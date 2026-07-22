"""
semantic_cache.py
=================
PHASE 4 v3 — Deterministic Semantic Cache (Dual-Lock: Entity + Vector).

This module implements a two-factor semantic cache:

  Factor 1 — The Lock (deterministic string):
    A "lock" is a combination of the main topic (Entity) and the specific
    details (Modifiers: version, OS, action). It is built by the RuleExtractor
    which uses flashtext keyword matching + regex rules. If the topic entity
    is unknown or ambiguous, NO lock is created and the cache is bypassed.

  Factor 2 — The Vector (cosine similarity):
    The current user message is embedded via the shared ONNX engine and
    compared against cached vectors. A cached entry is only returned if
    BOTH the lock string matches exactly AND the vector similarity >= 0.88.

This dual-lock prevents dangerous collisions (e.g. "install Python 3.14"
should NOT match "install Python 3.15" — different version modifier).

We use standard Python logging for observability: every cache hit, miss,
and rejected unknown entity is logged.
"""

import re
import logging
import time
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict

import numpy as np
import faiss
from flashtext import KeywordProcessor

from guardrail import embed_text, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
# We create a dedicated logger for the semantic cache so that cache events
# (hits, misses, entity rejections) are clearly visible in the log stream.
# ---------------------------------------------------------------------------
logger = logging.getLogger("SemanticCache")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
CACHE_MAX_SIZE: int = 10000
SIMILARITY_THRESHOLD: float = 0.88


# ---------------------------------------------------------------------------
# RULE EXTRACTOR — The Gazetteer + Modifier Engine
# ---------------------------------------------------------------------------
# This class reads the conversation history and extracts a "Lock".
# A Lock is a string that uniquely identifies the topic + modifiers of a
# user's question. It has two parts:
#
#   1. Entity: The main subject (e.g. "python", "docker").
#               Found via flashtext keyword matching against a seed dictionary.
#   2. Modifiers: Extra details (version, OS, action).
#                  Found via regex rules applied to the current message.
#
# If the entity is unknown (not in the dictionary) or ambiguous (multiple
# entities found in the conversation history), we return None — the cache
# is bypassed because we cannot safely build a lock.
# ---------------------------------------------------------------------------

class RuleExtractor:
    """
    STORY: This class reads the conversation and extracts a 'Lock'.
    A Lock is a combination of the main topic (Entity) and specific details
    (Modifiers). If we do not know the topic, we refuse to make a Lock.
    """

    def __init__(self):
        # Step 1: Setup the Entity Dictionary (The Gazetteer)
        # We use flashtext's KeywordProcessor for fast, case-insensitive
        # keyword matching against our seed list of known entities.
        logger.info("RuleExtractor: Initialising entity gazetteer ...")
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.seed_entities = [
            "python", "java", "docker", "kubernetes", "react",
            "node", "postgres",
        ]
        self.keyword_processor.add_keywords_from_list(self.seed_entities)
        logger.info(
            "RuleExtractor: Gazetteer loaded with %d entities.",
            len(self.seed_entities),
        )

        # Step 2: Setup the Modifiers (Regex rules)
        # These patterns extract version numbers, operating system names,
        # and action verbs from the user's message.
        self.version_rule = re.compile(
            r'(?:v|version)?\s*(\d+\.\d+(?:\.\d+)?)', re.IGNORECASE,
        )
        self.os_rule = re.compile(
            r'\b(windows|mac|linux|ubuntu|alpine)\b', re.IGNORECASE,
        )
        self.action_rule = re.compile(
            r'\b(install|uninstall|start|stop|deploy|remove)\b', re.IGNORECASE,
        )

    def _extract_entity_from_message(self, text: str) -> Optional[str]:
        """
        Extract a single entity from a message text using flashtext.

        Returns the first matched entity, or None if no entity is found.
        """
        matches = self.keyword_processor.extract_keywords(text)
        if not matches:
            return None
        # Return the first match (flashtext returns a list of matched strings).
        return matches[0]

    def _extract_modifiers(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract version, OS, and action modifiers from a message text.

        Returns a dict with keys: version, os, action.
        Each value is the matched string or None if not found.
        """
        version_match = self.version_rule.search(text)
        os_match = self.os_rule.search(text)
        action_match = self.action_rule.search(text)

        return {
            "version": version_match.group(1) if version_match else None,
            "os": os_match.group(1).lower() if os_match else None,
            "action": action_match.group(1).lower() if action_match else None,
        }

    def generate_lock(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Generate a deterministic lock string from the conversation history.

        STORY LOGIC:
        1. Find the current user message (the last message in the list).
        2. Ask the keyword_processor to find entities in this current message.
        3. If no entities are found in the current message:
           - Look backward at the previous user messages in the history.
           - If you find EXACTLY ONE entity in the history, use it.
           - If you find ZERO or MORE THAN ONE, return None (We are confused,
             so we give up — better to miss the cache than return wrong data).
        4. If we have a single valid entity, search the current message for
           our Regex Modifiers (version, OS, action).
        5. Combine them into a single pipe-delimited string.
           Example: "python|3.14|windows|install"
        6. Return this lock string.

        Parameters
        ----------
        messages : list of dict
            The conversation history. Each dict has "role" and "content" keys.

        Returns
        -------
        str or None
            The lock string, or None if the entity is unknown/ambiguous.
        """
        if not messages:
            logger.warning("generate_lock: Empty messages list.")
            return None

        # Step 1: Find the current user message (the last message in the list).
        current_message_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_message_text = msg.get("content", "")
                break

        if not current_message_text:
            logger.warning("generate_lock: No user message found.")
            return None

        # Step 2: Try to find an entity in the current message.
        current_entity = self._extract_entity_from_message(current_message_text)

        # Step 3: If no entity in the current message, look backward through
        # the conversation history for a previous user message with an entity.
        if current_entity is None:
            logger.info(
                "generate_lock: No entity in current message. "
                "Scanning history ...",
            )

            # Collect all entities found in previous user messages.
            historical_entities = []
            for msg in messages:
                if msg.get("role") == "user":
                    entity = self._extract_entity_from_message(
                        msg.get("content", ""),
                    )
                    if entity is not None:
                        historical_entities.append(entity)

            # If we found exactly one historical entity, use it.
            # If zero or more than one, we cannot disambiguate safely.
            if len(historical_entities) == 1:
                chosen_entity = historical_entities[0]
                logger.info(
                    "generate_lock: Using historical entity '%s'.",
                    chosen_entity,
                )
            else:
                logger.warning(
                    "generate_lock: Found %d historical entities — "
                    "cannot disambiguate. Aborting lock.",
                    len(historical_entities),
                )
                return None
        else:
            chosen_entity = current_entity

        # Step 4: Extract modifiers (version, OS, action) from the current
        # message only. We do NOT look for modifiers in historical messages.
        modifiers = self._extract_modifiers(current_message_text)

        # Step 5: Build the lock string as "entity|version|os|action".
        # Missing modifiers become empty strings.
        lock_parts = [
            chosen_entity,
            modifiers["version"] or "",
            modifiers["os"] or "",
            modifiers["action"] or "",
        ]
        lock_string = "|".join(lock_parts)

        logger.info(
            "generate_lock: Created lock '%s' from message: %.50s",
            lock_string,
            current_message_text,
        )
        return lock_string


# ---------------------------------------------------------------------------
# DETERMINISTIC SEMANTIC CACHE — The Vault
# ---------------------------------------------------------------------------
# This is the vault. It checks if we have seen a request before by requiring
# TWO keys to open:
#
#   Key 1 — The exact dictionary 'Lock' string (from RuleExtractor).
#   Key 2 — A mathematical vector similarity score >= 0.88.
#
# Both must match for a cache hit. This prevents version collisions and
# pronoun-based false positives.
# ---------------------------------------------------------------------------

class DeterministicSemanticCache:
    """
    STORY: This is the vault. It checks if we have seen a request before.
    It requires TWO keys to open:
    1. The exact dictionary 'Lock' string.
    2. A mathematical vector similarity score >= 0.88.
    """

    def __init__(self, embedding_function=embed_text):
        logger.info(
            "DeterministicSemanticCache: Initialising with "
            "threshold=%.2f, max_size=%d ...",
            SIMILARITY_THRESHOLD,
            CACHE_MAX_SIZE,
        )

        self.extractor = RuleExtractor()
        self.embedding_function = embedding_function
        self.threshold = SIMILARITY_THRESHOLD
        self.max_size = CACHE_MAX_SIZE

        # FAISS IndexIDMap wrapping a flat inner-product index.
        # Since all vectors are L2-normalised, inner product = cosine sim.
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatIP(EMBEDDING_DIM),
        )

        # LRU memory: { faiss_id : {"lock": lock_string, "response": dict} }
        self._lru_store: OrderedDict = OrderedDict()
        self._next_id: int = 0

        logger.info("DeterministicSemanticCache: Ready (0 vectors).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_one(self):
        """Remove the oldest (least recently used) entry from the cache."""
        if not self._lru_store:
            return
        oldest_id, _ = self._lru_store.popitem(last=False)
        try:
            self.index.remove_ids(np.array([oldest_id], dtype=np.int64))
        except Exception:
            logger.exception("FAISS removal failed during eviction.")
        logger.debug("Evicted cache entry id=%d.", oldest_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_cache(
        self, messages: List[Dict[str, str]],
    ) -> Optional[Dict]:
        """
        Check whether a semantically similar query exists in the cache.

        STORY LOGIC:
        1. Ask the RuleExtractor to generate a lock for these messages.
        2. If the lock is None:
           - Log "Cache aborted: Unknown or ambiguous entity."
           - Return None (Cache Miss).
        3. If we have a lock, convert ONLY the current user message into a
           math vector using self.embedding_function.
        4. Search the FAISS index for the top 5 closest vectors.
        5. For every result with similarity >= self.threshold (0.88):
           - Check our OrderedDict. Does the saved lock string EXACTLY match
             our current lock string?
           - If YES, log "Cache Hit", update the LRU, and return the saved
             response.
        6. If no matches pass both tests, return None (Cache Miss).

        Parameters
        ----------
        messages : list of dict
            The full conversation history.

        Returns
        -------
        dict or None
            The cached response dict, or None for a cache miss.
        """
        # Step 1: Generate the deterministic lock.
        lock_string = self.extractor.generate_lock(messages)

        # Step 2: If lock is None, we cannot safely consult the cache.
        if lock_string is None:
            logger.info(
                "Cache check aborted: Unknown or ambiguous entity.",
            )
            return None

        # Step 3: Extract the current user message and embed it.
        current_user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_user_text = msg.get("content", "")
                break

        if not current_user_text:
            logger.warning("Cache check: No user message to embed.")
            return None

        try:
            query_vector = self.embedding_function(current_user_text)
        except Exception:
            logger.exception(
                "Cache check: Embedding failed — failing open to miss.",
            )
            return None

        # Step 4: Search FAISS for the top 5 nearest neighbours.
        if self.index.ntotal == 0:
            logger.debug("Cache check: Index is empty — miss.")
            return None

        try:
            distances, ids = self.index.search(query_vector, k=5)
        except Exception:
            logger.exception(
                "Cache check: FAISS search failed — failing open to miss.",
            )
            return None

        # Step 5: Check each candidate for BOTH similarity AND lock match.
        for rank in range(len(ids[0])):
            similarity = float(distances[0][rank])
            similarity = max(0.0, min(1.0, similarity))

            if similarity < self.threshold:
                # Candidates are sorted by descending distance, so once we
                # fall below threshold, all remaining candidates will also
                # be below it.
                break

            matched_id = int(ids[0][rank])
            entry = self._lru_store.get(matched_id)

            if entry is None:
                # FAISS has a vector but our LRU store does not have the
                # metadata — this should not happen, but if it does we skip.
                continue

            saved_lock = entry.get("lock", "")
            saved_response = entry.get("response")

            # Does the saved lock EXACTLY match our current lock?
            if saved_lock == lock_string and saved_response is not None:
                # Cache Hit!
                logger.info(
                    "Cache HIT: lock='%s', similarity=%.4f, id=%d.",
                    lock_string, similarity, matched_id,
                )
                # Move to end of LRU (mark as recently used).
                self._lru_store.move_to_end(matched_id)
                return saved_response

        # Step 6: No candidate passed both checks.
        logger.info(
            "Cache MISS: lock='%s' (no matching entry).",
            lock_string,
        )
        return None

    async def insert(self, messages: List[Dict[str, str]], response: dict):
        """
        Insert a new query-response pair into the cache.

        STORY LOGIC:
        1. Generate the lock string from the messages.
        2. If lock is None, do not cache (unknown entity — safer to skip).
        3. Embed the current user message.
        4. If the LRU is full (10,000 entries), evict the oldest item.
        5. Add the vector to FAISS and store the metadata in the LRU dict.

        Parameters
        ----------
        messages : list of dict
            The conversation history.
        response : dict
            The response to cache.
        """
        # Step 1: Generate the lock.
        lock_string = self.extractor.generate_lock(messages)
        if lock_string is None:
            logger.info(
                "Cache insert skipped: No lock (unknown/ambiguous entity).",
            )
            return

        # Step 2: Extract the current user message and embed it.
        current_user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                current_user_text = msg.get("content", "")
                break

        if not current_user_text:
            logger.warning("Cache insert: No user message to embed.")
            return

        try:
            query_vector = self.embedding_function(current_user_text)
        except Exception:
            logger.exception("Cache insert: Embedding failed — skipping.")
            return

        # Step 3: Evict if at capacity.
        if len(self._lru_store) >= self.max_size:
            self._evict_one()

        # Step 4: Assign an ID and add to FAISS.
        entry_id = self._next_id
        self._next_id += 1

        try:
            self.index.add_with_ids(
                query_vector,
                np.array([entry_id], dtype=np.int64),
            )
        except Exception:
            logger.exception("Cache insert: FAISS add failed — skipping.")
            return

        # Step 5: Store metadata in the LRU dict.
        self._lru_store[entry_id] = {
            "lock": lock_string,
            "response": response,
        }

        logger.info(
            "Cache inserted: id=%d, lock='%s'. Total size=%d.",
            entry_id, lock_string, len(self._lru_store),
        )

    @property
    def size(self) -> int:
        """Return the current number of entries in the cache."""
        return len(self._lru_store)

    def clear(self):
        """Reset the cache entirely."""
        dim = EMBEDDING_DIM
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._lru_store.clear()
        self._next_id = 0
        logger.info("Cache cleared.")


# ---------------------------------------------------------------------------
# GLOBAL INSTANCE
# ---------------------------------------------------------------------------
logger.info("Creating global DeterministicSemanticCache instance ...")
semantic_cache = DeterministicSemanticCache()
logger.info("Semantic Cache ready.")
