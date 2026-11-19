"""
scripts/derive_engine_power.py
==============================
Derives mean GPU power draw (watts) per engine from Phase 8 measured run logs.
Reads 1 Hz nvidia-smi power.draw samples from archive/phase8/power_draw_<run_id>.log,
maps each run_id to its engine via archive/phase8/telemetry_<run_id>.jsonl run_metadata,
and outputs the pooled means and sample counts.
"""

from __future__ import annotations

import json
from pathlib import Path


def derive_engine_power(archive_dir: Path | str) -> dict[str, tuple[float, int]]:
    """
    Computes mean power draw in watts and sample counts per engine.

    Inputs:
    - archive_dir: Directory containing power_draw_*.log and telemetry_*.jsonl files.

    Returns:
    - Dict mapping engine_id to (mean_watts, sample_count).
    """
    base = Path(archive_dir)
    power_files = sorted(base.glob("power_draw_*.log"))

    engine_samples: dict[str, list[float]] = {}

    for power_path in power_files:
        run_id = power_path.stem.replace("power_draw_", "")
        telem_path = base / f"telemetry_{run_id}.jsonl"
        if not telem_path.is_file():
            continue

        with open(telem_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                continue
            metadata = json.loads(first_line)
            engine = metadata["run_metadata"]["engine"]

        samples: list[float] = []
        with open(power_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.endswith(" W"):
                    val = float(line.replace(" W", "").strip())
                    samples.append(val)

        engine_samples.setdefault(engine, []).extend(samples)

    results: dict[str, tuple[float, int]] = {}
    for engine, samples in sorted(engine_samples.items()):
        mean_val = sum(samples) / len(samples) if samples else 0.0
        results[engine] = (mean_val, len(samples))

    return results


def main() -> None:
    """Entry point: runs derivation from archive/phase8 and prints formatted results."""
    repo_root = Path(__file__).resolve().parent.parent
    archive_dir = repo_root / "archive" / "phase8"

    results = derive_engine_power(archive_dir)
    print("=== Derived Engine Power Draws (Phase 8 Archive) ===")
    for engine, (mean_watts, count) in results.items():
        print(f"{engine}: {mean_watts:.2f} W (samples: {count})")


if __name__ == "__main__":
    main()
