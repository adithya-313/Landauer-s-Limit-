"""
semantic_cache.py
=================
PHASE 4 — Semantic Cache with FAISS IndexIDMap and LRU Eviction.

This module provides a semantic cache that stores question → answer pairs by
their embedding vectors. When a new user message arrives, we embed it and
check whether a semantically similar question has already been answered. If
the cosine similarity to a cached entry exceeds the threshold (0.95), we
return the cached response instead of calling the LLM.

Multi-turn handling:
  The cache key for multi-turn conversations is formed by concatenating the
  last assistant response (if any) with the current user question. This
  ensures that the same question asked in a different conversational context
  is treated as a distinct cache entry.

Key design decisions (locked):
  - Vector store: FAISS IndexIDMap (allows mapping vector IDs to metadata).
  - Eviction policy: LRU via collections.OrderedDict (max 10 000 entries).
  - Similarity threshold: 0.95 (marked as "pending empirical tuning").
"""

import time
from typing import Tuple
from collections import OrderedDict

import numpy as np
import faiss

from guardrail import embed_text, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Maximum number of entries in the cache before LRU eviction kicks in.
CACHE_MAX_SIZE: int = 10000

# Cosine similarity threshold. A query must exceed this to count as a cache
# hit. Set to 0.95 initially — tune empirically based on real traffic.
SIMILARITY_THRESHOLD: float = 0.95


# ---------------------------------------------------------------------------
# SEMANTIC CACHE CLASS
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Semantic cache using FAISS IndexIDMap + LRU eviction.

    What this cache does:
      - Stores (text, response) pairs keyed by their embedding vector.
      - On check_cache(): embeds the query, searches FAISS for the nearest
        neighbour, and returns the cached response if similarity >= threshold.
      - On insert(): embeds the new text and stores it in FAISS. If the cache
        is full, the least-recently-used entry is evicted first.

    Usage:
        cache = SemanticCache()
        hit, response, sim = cache.check_cache("What is Python?")
        if not hit:
            response = llm_call("What is Python?")
            cache.insert("What is Python?", response)
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        print(
            f"[SemanticCache] Initialising with max_size={max_size}, "
            f"threshold={SIMILARITY_THRESHOLD} ...",
        )

        self.max_size = max_size

        # --- FAISS IndexIDMap for vector storage ---
        # We wrap a flat IP (inner product) index with an IDMap so that we
        # can assign arbitrary integer IDs to vectors and later remove them
        # by ID during LRU eviction.
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatIP(EMBEDDING_DIM),
        )

        # --- LRU tracking via OrderedDict ---
        # Each entry: id -> (cache_key_text, response, timestamp)
        # Whenever an entry is accessed (check_cache hit), we move it to the
        # end of the OrderedDict to mark it as recently used.
        self._lru: OrderedDict = OrderedDict()

        # Auto-incrementing ID counter.
        self._next_id: int = 0

        print(f"[SemanticCache] Ready. {self.index.ntotal} vectors stored.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cache_key(self, user_message: str, context: str = "") -> str:
        """
        Build the string that gets embedded for cache lookup.

        For single-turn queries:
            cache_key = user_message

        For multi-turn queries:
            cache_key = context + " | " + user_message

        Parameters
        ----------
        user_message : str
            The current user message.
        context : str, optional
            The last assistant response (used for multi-turn context).

        Returns
        -------
        str
            The cache key string to embed.
        """
        if context:
            return f"{context} | {user_message}"
        return user_message

    def _evict_one(self):
        """
        Remove the least-recently-used entry from the cache.

        We pop the first (oldest) item from the OrderedDict and remove its
        corresponding vector from the FAISS index.
        """
        if not self._lru:
            return

        # Pop the oldest entry (first item in OrderedDict).
        oldest_id, _ = self._lru.popitem(last=False)

        # Remove from FAISS index by ID.
        try:
            self.index.remove_ids(np.array([oldest_id], dtype=np.int64))
        except Exception:
            pass  # If removal fails, the index may already be inconsistent.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_cache(
        self, user_message: str, context: str = "",
    ) -> Tuple[bool, str, float]:
        """
        Check whether a semantically similar query exists in the cache.

        Parameters
        ----------
        user_message : str
            The current user message.
        context : str, optional
            The last assistant response for multi-turn context.

        Returns
        -------
        (hit, cached_response, similarity)
            hit : True if a similar cached entry was found.
            cached_response : The cached response string (empty if miss).
            similarity : Cosine sim to the nearest cached entry (0.0 if miss).
        """
        # Build the cache key and embed it.
        cache_key = self._build_cache_key(user_message, context)
        # Already L2-normalised, shape (1, 384).
        query_vec = embed_text(cache_key)

        # If the index is empty, there is nothing to match against.
        if self.index.ntotal == 0:
            return False, "", 0.0

        # Search for the single nearest neighbour.
        distances, ids = self.index.search(query_vec, k=1)
        similarity = float(distances[0][0])

        # Clamp similarity to [0, 1].
        similarity = max(0.0, min(1.0, similarity))

        # If similarity is below threshold, it's a cache miss.
        if similarity < SIMILARITY_THRESHOLD:
            return False, "", similarity

        # Cache hit! Look up the ID in our LRU dict.
        matched_id = int(ids[0][0])
        entry = self._lru.get(matched_id)

        if entry is None:
            # The ID exists in FAISS but our LRU dict is out of sync.
            # Treat as a miss to be safe.
            return False, "", similarity

        cached_text, cached_response, _ = entry

        # Move to end of LRU (mark as recently used).
        self._lru.move_to_end(matched_id)

        return True, cached_response, similarity

    def insert(self, user_message: str, response: str, context: str = ""):
        """
        Insert a new query → response pair into the cache.

        Parameters
        ----------
        user_message : str
            The user message that was asked.
        response : str
            The assistant response to cache.
        context : str, optional
            The last assistant response for multi-turn context.
        """
        # Build the cache key and embed it.
        cache_key = self._build_cache_key(user_message, context)
        query_vec = embed_text(cache_key)  # Already L2-normalised.

        # Evict the LRU entry if at capacity.
        if len(self._lru) >= self.max_size:
            self._evict_one()

        # Assign a new ID and add to FAISS.
        entry_id = self._next_id
        self._next_id += 1

        self.index.add_with_ids(
            query_vec, np.array([entry_id], dtype=np.int64),
        )

        # Store metadata in the LRU dict.
        self._lru[entry_id] = (cache_key, response, time.time())

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._lru)

    def clear(self):
        """Reset the cache entirely."""
        dim = EMBEDDING_DIM
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._lru.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# GLOBAL INSTANCE (singleton for import convenience)
# ---------------------------------------------------------------------------

print("[SemanticCache] Creating global instance ...")
semantic_cache = SemanticCache()
print(f"[SemanticCache] Ready. Size = {semantic_cache.size}.")
