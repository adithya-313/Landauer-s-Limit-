"""Generate markdown reference tables for reproducibility documentation.

This script parses REQUIRED_MANIFEST_FIELDS from run_with_manifest.py,
model definitions from models/manifest.yaml, and primary targets from Makefile,
outputting deterministic Markdown tables.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Tuple
import yaml

from run_with_manifest import REQUIRED_MANIFEST_FIELDS

FIELD_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "schema_version": ("int", "Schema specification version (currently 1)."),
    "run_id": ("str (UUID)", "Unique execution UUID identifying telemetry and logs."),
    "started_at": ("str (ISO-8601 UTC)", "Benchmark execution initiation timestamp."),
    "finished_at": ("str (ISO-8601 UTC)", "Benchmark completion timestamp."),
    "duration_seconds": ("int", "Target benchmark duration in seconds."),
    "engine": ("str", "Inference engine key (e.g. llamacpp_local, vllm_local)."),
    "quantization": ("str", "Quantization format specification (e.g. q4, awq)."),
    "concurrency": ("int", "Number of concurrent simulated client requests."),
    "injected_failure": ("str | None", "Active failure injection scenario name, if any."),
    "git_commit": ("str", "Exact 40-character Git SHA at time of benchmark run."),
    "git_dirty": ("bool", "Flag indicating whether uncommitted git changes existed."),
    "lockfile_hashes": ("dict[str, str]", "SHA-256 digests of all dependency lockfiles."),
    "config_hashes": ("dict[str, str]", "SHA-256 digests of runtime config and tenant files."),
    "config_files_missing": ("list[str]", "List of optional configuration files that were missing."),
    "engine_model": ("dict", "Model metadata entry from models/manifest.yaml for the active engine."),
    "model_revisions": ("dict[str, str]", "Pinned model repository revisions (git commit SHAs)."),
    "routing_rule_set_version": ("str", "Active routing rule set version string from config.yaml."),
    "slo_version": ("str", "Active service level objective policy version from config.yaml."),
    "version_notes": ("str", "Reproducibility notes and assumptions recorded for this run."),
    "gpu_name": ("str | None", "GPU hardware name reported by nvidia-smi."),
    "gpu_driver": ("str | None", "GPU driver version string reported by nvidia-smi."),
    "cuda_version": ("str | None", "CUDA runtime / driver version from nvidia-smi."),
    "gpu_info_error": ("str | None", "Error diagnostics if nvidia-smi query failed."),
    "output_files": ("dict[str, str]", "SHA-256 hashes of telemetry and power logs."),
}

MAKEFILE_TARGET_DESCRIPTIONS: Dict[str, str] = {
    "all": "Run lint, typecheck, test-ci, and reproduce-report targets in sequence.",
    "reproduce-report": "Verify phase8 archive checksums and reproduce report.md.",
    "lint": "Run ruff code checks against all tracked repository modules.",
    "typecheck": "Run mypy static type checking against core repository modules.",
    "test-ci": "Execute full continuous integration pytest suite.",
    "test-gpu": "Execute GPU-dependent benchmark and adapter test suites.",
    "print-ci-files": "Print lists of files targeted for lint, typecheck, and CI tests.",
}


def generate_manifest_fields_table() -> str:
    """Generate Markdown table documenting all required run manifest fields.

    Returns:
        str: Formatted Markdown table.
    """
    lines: List[str] = [
        "| Field Name | Type | Description |",
        "| --- | --- | --- |",
    ]
    for field in REQUIRED_MANIFEST_FIELDS:
        type_str, desc = FIELD_DESCRIPTIONS.get(
            field, ("Any", "Recorded execution parameter.")
        )
        lines.append(f"| `{field}` | `{type_str}` | {desc} |")
    return "\n".join(lines)


def generate_models_table(manifest_path: Path = Path("models/manifest.yaml")) -> str:
    """Generate Markdown table documenting registered models from models/manifest.yaml.

    Args:
        manifest_path (Path): Path to models manifest YAML.

    Returns:
        str: Formatted Markdown table.
    """
    content = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    models = content.get("models", {})

    lines: List[str] = [
        "| Model ID | Source | Quantization | Verified | Note |",
        "| --- | --- | --- | --- | --- |",
    ]

    for model_id in sorted(models.keys()):
        data = models[model_id]
        source = data.get("source", "-")
        quant = data.get("quantization", "-")
        verified = "Yes" if data.get("verified") else "No"
        note = data.get("note", "-")
        lines.append(f"| `{model_id}` | `{source}` | `{quant}` | {verified} | {note} |")
    return "\n".join(lines)


def generate_makefile_targets_table(makefile_path: Path = Path("Makefile")) -> str:
    """Generate Markdown table documenting primary Makefile targets.

    Args:
        makefile_path (Path): Path to Makefile.

    Returns:
        str: Formatted Markdown table.
    """
    lines: List[str] = [
        "| Target | Description |",
        "| --- | --- |",
    ]
    for target in sorted(MAKEFILE_TARGET_DESCRIPTIONS.keys()):
        desc = MAKEFILE_TARGET_DESCRIPTIONS[target]
        lines.append(f"| `make {target}` | {desc} |")
    return "\n".join(lines)


def generate_all_tables() -> str:
    """Generate all reproducibility reference tables as a unified Markdown block.

    Returns:
        str: Markdown block containing all reference tables.
    """
    manifest_table = generate_manifest_fields_table()
    models_table = generate_models_table()
    makefile_table = generate_makefile_targets_table()

    block = (
        "### Run Manifest Fields\n\n"
        f"{manifest_table}\n\n"
        "### Registered Models\n\n"
        f"{models_table}\n\n"
        "### Primary Makefile Targets\n\n"
        f"{makefile_table}"
    )
    return block


def main() -> int:
    """CLI entrypoint printing generated tables to standard output."""
    sys.stdout.write(generate_all_tables() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
