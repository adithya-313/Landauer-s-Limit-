"""Test verification for docs/reproducibility.md generated tables."""

from pathlib import Path
import re

import scripts.gen_reproducibility_tables as gen_tables


def test_reproducibility_docs_tables_match_generator():
    """Verify that generated tables block in docs/reproducibility.md is up to date."""
    doc_path = Path("docs/reproducibility.md")
    assert doc_path.is_file(), "docs/reproducibility.md must exist"

    content = doc_path.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- generated:start -->\n(.*?)\n<!-- generated:end -->",
        content,
        re.DOTALL,
    )
    assert match is not None, "Generated marker comments missing in docs/reproducibility.md"

    doc_block = match.group(1).strip()
    generated_block = gen_tables.generate_all_tables().strip()

    # Normalizing line endings
    doc_normalized = doc_block.replace("\r\n", "\n")
    gen_normalized = generated_block.replace("\r\n", "\n")

    assert doc_normalized == gen_normalized, (
        "docs/reproducibility.md tables do not match scripts/gen_reproducibility_tables.py output. "
        "Re-run table generator to synchronize."
    )

