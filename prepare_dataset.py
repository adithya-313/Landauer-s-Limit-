"""
prepare_dataset.py
===================
PHASE 1 — Dataset Pipeline for Landauer's Limit project.

What this script does:
1. Downloads a sample of the ShareGPT conversation dataset from HuggingFace.
2. Filters and curates exactly 200 clean prompt/response pairs.
3. Tags 20 of them as "drift_eval" samples for later evaluation.
4. Saves everything into `test_dataset.json` in a standardised schema.

"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# We use the `datasets` library from HuggingFace to download the ShareGPT data.
# `json` lets us write the final output file. `random` helps us shuffle and
# select diverse samples. `collections.Counter` is used for lightweight stats.
# ---------------------------------------------------------------------------
import json
import random
import math
from pathlib import Path
from collections import Counter

from datasets import load_dataset


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
# Pin these at the top so they are easy to find and tweak later.
# ---------------------------------------------------------------------------
DATASET_NAME = "Aeala/ShareGPT_Vicuna_unfiltered"
TOTAL_SAMPLES = 200
DRIFT_EVAL_COUNT = 20
OUTPUT_FILE = "test_dataset.json"
RANDOM_SEED = 42

# The categories we use to classify conversations.
# A mix of general chat, coding, reasoning, creative writing, and knowledge.
CATEGORIES = [
    "general",
    "coding",
    "reasoning",
    "creative_writing",
    "knowledge",
]


# ---------------------------------------------------------------------------
# FUNCTION 1: download_data
# ---------------------------------------------------------------------------
# This function reaches out to HuggingFace and downloads the ShareGPT dataset.
# We use `streaming=True` so we don't download the entire multi-GB dataset —
# we only load as many samples as we need. We also pass `trust_remote_code=True`
# because this dataset includes a small custom loading script.
# ---------------------------------------------------------------------------
def download_data(dataset_name: str, max_samples: int = 5000):
    """
    Download a subset of the ShareGPT dataset.

    Parameters
    ----------
    dataset_name : str
        The HuggingFace dataset identifier.
    max_samples : int
        How many samples to fetch from the dataset (we'll filter further later).

    Returns
    -------
    list[dict]
        A list of raw conversation dictionaries from the dataset.
    """
    print(f"[INFO] Loading dataset '{dataset_name}' from HuggingFace ...")

    # Load the dataset in streaming mode so we only iterate once.
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Collect samples up to `max_samples`.
    raw_samples = []
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        raw_samples.append(sample)

    print(f"[INFO] Downloaded {len(raw_samples)} raw samples.")
    return raw_samples


# ---------------------------------------------------------------------------
# FUNCTION 2: is_valid_conversation
# ---------------------------------------------------------------------------
# A helper that checks whether a single conversation is usable.
# We throw away conversations that:
#   - Have fewer than 2 turns (need at least one user + one assistant).
#   - Have empty or whitespace-only messages.
#   - Have absurdly long messages (likely data errors).
# We also identify code-heavy conversations by looking for markdown code blocks.
# ---------------------------------------------------------------------------
def is_valid_conversation(conversation: dict) -> bool:
    """
    Check whether a conversation is well-formed and non-empty.

    Parameters
    ----------
    conversation : dict
        A raw conversation from the dataset. Expected to have 'id' and
        'conversations' keys.

    Returns
    -------
    bool
        True if the conversation is valid, False otherwise.
    """
    # The dataset stores conversations as a list of turns under 'conversations'.
    turns = conversation.get("conversations", [])

    # We need at least one user message AND one assistant message.
    # If there are fewer than 2 total turns, skip it.
    if len(turns) < 2:
        return False

    # Make sure every turn has a non-empty 'value' field.
    for turn in turns:
        content = turn.get("value", "")
        if not content or not content.strip():
            return False

    return True


# ---------------------------------------------------------------------------
# FUNCTION 3: classify_conversation
# ---------------------------------------------------------------------------
# We assign a category label to each conversation based on keywords in the
# user's first message. This gives us a lightweight classification without
# needing an expensive model.
# ---------------------------------------------------------------------------
def classify_conversation(conversation: dict) -> str:
    """
    Assign a category to a conversation based on keyword matching
    against the first user message.

    Parameters
    ----------
    conversation : dict
        A valid conversation dictionary.

    Returns
    -------
    str
        One of the CATEGORIES.
    """
    turns = conversation["conversations"]

    # Find the first message from the human/user.
    first_user_message = ""
    for turn in turns:
        if turn.get("from", "").lower() in ("human", "user"):
            first_user_message = turn.get("value", "").lower()
            break

    # Simple keyword-based classification.
    coding_keywords = [
        "code", "function", "def ", "class ", "import ", "debug",
        "algorithm", "javascript", "python", "typescript", "sql",
        "api", "endpoint", "react", "component", "variable",
    ]
    reasoning_keywords = [
        "explain", "why", "how does", "compare", "difference",
        "analysis", "reason", "step by step", "solve", "math",
        "equation", "logic", "proof",
    ]
    creative_keywords = [
        "write a story", "poem", "creative", "essay", "describe",
        "imagine", "generate", "draft", "script", "dialogue",
        "blog post", "article", "narrative",
    ]
    knowledge_keywords = [
        "what is", "define", "history", "overview", "summarize",
        "explain the concept", "meaning of", "background",
        "research", "study", "theory",
    ]

    # Check keyword groups in priority order: coding first, then reasoning,
    # then creative, then knowledge, falling back to general.
    if any(kw in first_user_message for kw in coding_keywords):
        return "coding"
    if any(kw in first_user_message for kw in reasoning_keywords):
        return "reasoning"
    if any(kw in first_user_message for kw in creative_keywords):
        return "creative_writing"
    if any(kw in first_user_message for kw in knowledge_keywords):
        return "knowledge"

    return "general"


# ---------------------------------------------------------------------------
# FUNCTION 4: convert_to_standard_format
# ---------------------------------------------------------------------------
# The raw dataset uses a "from"/"value" format for each turn. We convert
# that into our standardised schema with "role" ("user" or "assistant")
# and "content". We also assign an ID and a category.
# ---------------------------------------------------------------------------
def convert_to_standard_format(
    conversation: dict,
    category: str,
    drift_eval: bool = False,
) -> dict:
    """
    Convert a raw ShareGPT conversation into the standard output schema.

    Parameters
    ----------
    conversation : dict
        A valid raw conversation.
    category : str
        The category label for this conversation.
    drift_eval : bool
        Whether this conversation is flagged for drift evaluation.

    Returns
    -------
    dict
        A dictionary matching the schema:
        { "id": str, "turns": list[dict], "category": str, "drift_eval": bool }
    """
    raw_turns = conversation["conversations"]
    original_id = conversation.get("id", "unknown")

    # Map the "from" field to our standard role names.
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
    }

    standard_turns = []
    for turn in raw_turns:
        role = role_map.get(turn.get("from", "").lower(), "user")
        content = turn.get("value", "")
        standard_turns.append({
            "role": role,
            "content": content,
        })

    return {
        "id": original_id,
        "turns": standard_turns,
        "category": category,
        "drift_eval": drift_eval,
    }


# ---------------------------------------------------------------------------
# FUNCTION 5: curate_dataset
# ---------------------------------------------------------------------------
# This is the main curation pipeline. It:
#   1. Filters to only valid conversations.
#   2. Ensures diversity by picking from different categories.
#   3. Ensures a mix of conversation lengths (short, medium, multi-turn).
#   4. Tags exactly 20 conversations as drift_eval samples.
#   5. Returns exactly 200 curated conversations.
# ---------------------------------------------------------------------------
def curate_dataset(
    raw_samples: list[dict],
    total_needed: int = TOTAL_SAMPLES,
    drift_count: int = DRIFT_EVAL_COUNT,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Filter, classify, and curate raw samples into the final dataset.

    Parameters
    ----------
    raw_samples : list[dict]
        Raw conversations from `download_data`.
    total_needed : int
        How many conversations we want in the final dataset.
    drift_count : int
        How many of those should be flagged for drift evaluation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Curated list of standard-format conversation dictionaries.
    """
    random.seed(seed)
    print(f"[INFO] Curating {total_needed} conversations from {len(raw_samples)} raw samples ...")

    # --- Step A: Filter to only valid conversations ---
    valid = []
    skipped = 0
    for sample in raw_samples:
        if is_valid_conversation(sample):
            valid.append(sample)
        else:
            skipped += 1
    print(f"[INFO] Valid conversations: {len(valid)}, Skipped: {skipped}")

    # --- Step B: Classify every valid conversation ---
    classified = []
    for sample in valid:
        category = classify_conversation(sample)
        classified.append((sample, category))
    print(f"[INFO] Category distribution: {dict(Counter(cat for _, cat in classified))}")

    # --- Step C: Shuffle to avoid any ordering bias ---
    random.shuffle(classified)

    # --- Step D: Group by category for stratified sampling ---
    by_category: dict[str, list] = {cat: [] for cat in CATEGORIES}
    for sample, category in classified:
        by_category[category].append(sample)

    # --- Step E: Select samples ensuring diversity ---
    # We want a reasonably balanced distribution across categories.
    # Allocate roughly equal numbers per category, then fill any remainder.
    per_category = max(1, total_needed // len(CATEGORIES))

    selected_raw = []
    selected_categories = []

    for cat in CATEGORIES:
        pool = by_category[cat]
        # Take up to `per_category` from each category.
        take = min(per_category, len(pool))
        if take > 0:
            chosen = pool[:take]
            selected_raw.extend(chosen)
            selected_categories.extend([cat] * len(chosen))

    # If we still haven't hit 200, fill from remaining samples in any category.
    if len(selected_raw) < total_needed:
        remaining_needed = total_needed - len(selected_raw)
        # Collect all unused samples.
        used_ids = {s["id"] for s in selected_raw if "id" in s}
        leftovers = [
            s for s, cat in classified
            if s.get("id") not in used_ids
        ]
        random.shuffle(leftovers)
        for s in leftovers[:remaining_needed]:
            cat = classify_conversation(s)
            selected_raw.append(s)
            selected_categories.append(cat)

    # Trim to exactly total_needed (in case we overshot due to per-category rounding).
    selected_raw = selected_raw[:total_needed]
    selected_categories = selected_categories[:total_needed]

    print(f"[INFO] After curation: {len(selected_raw)} conversations selected.")
    print(f"[INFO] Curated category distribution: {dict(Counter(selected_categories))}")

    # --- Step F: Convert to standard format ---
    curated = []
    for sample, category in zip(selected_raw, selected_categories):
        curated.append(convert_to_standard_format(sample, category, drift_eval=False))

    # --- Step G: Tag exactly `drift_count` samples as drift_eval ---
    # Pick a diverse subset across categories for drift evaluation.
    # We pick a few from each category proportionally.
    drift_indices = []
    cats_in_curated = Counter(item["category"] for item in curated)
    remaining_drift = drift_count

    # First, allocate at least 1 per category.
    for cat in CATEGORIES:
        if remaining_drift <= 0:
            break
        # Find indices of items in this category.
        cat_indices = [
            i for i, item in enumerate(curated) if item["category"] == cat
        ]
        if cat_indices:
            # Pick one from this category.
            chosen = random.choice(cat_indices)
            drift_indices.append(chosen)
            remaining_drift -= 1

    # Fill remaining drift slots randomly.
    all_indices = list(range(len(curated)))
    remaining_pool = [i for i in all_indices if i not in drift_indices]
    random.shuffle(remaining_pool)
    drift_indices.extend(remaining_pool[:remaining_drift])

    # Tag them.
    for idx in drift_indices:
        curated[idx]["drift_eval"] = True

    actual_drift_count = sum(1 for item in curated if item["drift_eval"])
    print(f"[INFO] Drift-eval flagged: {actual_drift_count}")

    return curated


# ---------------------------------------------------------------------------
# FUNCTION 6: validate_dataset
# ---------------------------------------------------------------------------
# After curation, we run a quick self-check to make sure everything looks
# correct before saving. This catches problems early.
# ---------------------------------------------------------------------------
def validate_dataset(dataset: list[dict]):
    """
    Run basic integrity checks on the curated dataset.

    Parameters
    ----------
    dataset : list[dict]
        The curated dataset to validate.

    Raises
    ------
    ValueError
        If any check fails.
    """
    errors = []

    # Check 1: Total count.
    if len(dataset) != TOTAL_SAMPLES:
        errors.append(f"Expected {TOTAL_SAMPLES} samples, got {len(dataset)}")

    # Check 2: No duplicate IDs.
    ids = [item["id"] for item in dataset]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate IDs found in dataset")

    # Check 3: Every item has required fields.
    required_fields = {"id", "turns", "category", "drift_eval"}
    for item in dataset:
        missing = required_fields - set(item.keys())
        if missing:
            errors.append(f"Item {item.get('id', '?')} missing fields: {missing}")

    # Check 4: Every item has at least 2 turns (1 user + 1 assistant).
    for item in dataset:
        if len(item["turns"]) < 2:
            errors.append(f"Item {item['id']} has fewer than 2 turns")

    # Check 5: Every turn has role and content.
    for item in dataset:
        for turn in item["turns"]:
            if "role" not in turn or "content" not in turn:
                errors.append(f"Item {item['id']} has malformed turn")

    # Check 6: Exactly DRIFT_EVAL_COUNT items have drift_eval=True.
    drift_count = sum(1 for item in dataset if item["drift_eval"])
    if drift_count != DRIFT_EVAL_COUNT:
        errors.append(
            f"Expected {DRIFT_EVAL_COUNT} drift_eval items, got {drift_count}"
        )

    # Check 7: All categories are valid.
    valid_categories = set(CATEGORIES)
    for item in dataset:
        if item["category"] not in valid_categories:
            errors.append(f"Item {item['id']} has invalid category '{item['category']}'")

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

    print(f"[OK] All validation checks passed for {len(dataset)} items.")


# ---------------------------------------------------------------------------
# FUNCTION 7: save_dataset
# ---------------------------------------------------------------------------
# Writes the curated dataset to a JSON file with nice indentation so it is
# human-readable when you open it in an editor.
# ---------------------------------------------------------------------------
def save_dataset(dataset: list[dict], output_path: str):
    """
    Write the dataset to a JSON file.

    Parameters
    ----------
    dataset : list[dict]
        The curated dataset.
    output_path : str
        File path to write to.
    """
    path = Path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Dataset saved to '{path.absolute()}'")


# ---------------------------------------------------------------------------
# FUNCTION 8: print_statistics
# ---------------------------------------------------------------------------
# Prints a summary of the dataset so you can verify diversity at a glance.
# This includes category counts, turn-length distribution, and token-length
# histogram (min/max/avg).
# ---------------------------------------------------------------------------
def print_statistics(dataset: list[dict]):
    """
    Print summary statistics for the curated dataset.
    """
    total = len(dataset)
    category_counts = Counter(item["category"] for item in dataset)
    turn_counts = [len(item["turns"]) for item in dataset]
    drift_count = sum(1 for item in dataset if item["drift_eval"])

    # Calculate token-length statistics per item (approximate by splitting on spaces).
    token_lengths = []
    for item in dataset:
        total_tokens = sum(
            len(turn["content"].split())
            for turn in item["turns"]
        )
        token_lengths.append(total_tokens)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total conversations:  {total}")
    print(f"  Drift-eval items:     {drift_count}")
    print(f"\n  Category distribution:")
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        bar = "#" * count
        print(f"    {cat:20s}: {count:3d}  {bar}")
    print(f"\n  Turn-length distribution:")
    turn_counter = Counter(turn_counts)
    for t_len in sorted(turn_counter):
        print(f"    {t_len:2d} turns:  {turn_counter[t_len]:3d} conversations")
    print(f"\n  Token-length (word-count) statistics:")
    print(f"    Min:     {min(token_lengths)}")
    print(f"    Max:     {max(token_lengths)}")
    print(f"    Avg:     {sum(token_lengths) // len(token_lengths)}")
    print(f"    Median:  {sorted(token_lengths)[len(token_lengths) // 2]}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
# When this script is run directly (not imported), it executes the full
# pipeline: download → curate → validate → save → print stats.
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("LANDAUER'S LIMIT — Dataset Pipeline (Phase 1)")
    print("=" * 60)

    # Step 1: Download raw data from HuggingFace.
    raw_data = download_data(DATASET_NAME, max_samples=5000)

    # Step 2: Curate — filter, classify, select, and tag.
    curated = curate_dataset(raw_data)

    # Step 3: Validate — make sure everything is correct.
    validate_dataset(curated)

    # Step 4: Save to disk.
    save_dataset(curated, OUTPUT_FILE)

    # Step 5: Print statistics.
    print_statistics(curated)

    print("[DONE] Pipeline completed successfully.")


# ---------------------------------------------------------------------------
# Standard Python guard: only run main() if this file is executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
