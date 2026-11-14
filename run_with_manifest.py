"""Execution wrapper around orchestrator that builds, writes, and stores run manifests."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import asyncpg

import model_manifest
from scripts.verify_models import verify_single_model_files

ENGINE_MODEL_IDS: Dict[str, str] = {
    "vllm_local": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "llamacpp_local": "ollama/qwen2.5:1.5b",
    "colab_cloud": "Qwen/Qwen2.5-1.5B-Instruct",
    "custom_runtime": "custom_runtime",
}

REQUIRED_MANIFEST_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "run_id",
    "started_at",
    "finished_at",
    "engine",
    "quantization",
    "concurrency",
    "duration_seconds",
    "injected_failure",
    "git_commit",
    "git_dirty",
    "lockfile_hashes",
    "config_hashes",
    "config_files_missing",
    "engine_model",
    "model_revisions",
    "routing_rule_set_version",
    "slo_version",
    "version_notes",
    "gpu_name",
    "gpu_driver",
    "cuda_version",
    "gpu_info_error",
    "output_files",
)

DEFAULT_LOCKFILES: List[str] = [
    "requirements-gateway.lock",
    "requirements-ci.lock",
    "requirements-migrations.lock",
]

DEFAULT_CONFIG_FILES: List[str] = [
    "config.yaml",
    "tenants.yaml",
    "deploy/gateway-config/config.yaml",
    "deploy/gateway-config/tenants.yaml",
    "models/manifest.yaml",
]


def get_git_info() -> Tuple[str, bool]:
    """Retrieve current git commit hash and working tree dirty status.

    Purpose:
        Ensure run manifests record immutable repository revision and uncommitted state.

    Outputs:
        Tuple[str, bool]: (git_commit_sha, is_working_tree_dirty).

    Errors:
        RuntimeError: If git command fails or git repository is inaccessible.
    """
    res_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if res_commit.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {res_commit.stderr.strip()}")
    commit = res_commit.stdout.strip()

    res_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    if res_status.returncode != 0:
        raise RuntimeError(
            f"git status --porcelain failed: {res_status.stderr.strip()}"
        )
    dirty = bool(res_status.stdout.strip())
    return commit, dirty


def hash_files(paths: Sequence[Union[str, Path]]) -> Dict[str, str]:
    """Compute cryptographic SHA-256 digests for a list of file paths.

    Purpose:
        Record immutable artifact hashes in run manifest using streaming sha256_of_file.

    Inputs:
        paths (List[str | Path]): File paths to hash.

    Outputs:
        Dict[str, str]: Mapping of normalized POSIX file path string to lowercase sha256 hex.

    Errors:
        FileNotFoundError: If any specified path does not exist on disk.
    """
    out: Dict[str, str] = {}
    for p in paths:
        path_obj = Path(p)
        if not path_obj.is_file():
            raise FileNotFoundError(f"File not found for manifest hashing: {p}")
        out[str(p).replace("\\", "/")] = model_manifest.sha256_of_file(path_obj)
    return out


def get_gpu_info(
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Query local NVIDIA GPU driver, name, and CUDA version via nvidia-smi.

    Purpose:
        Record host hardware specifications for reproducibility without crashing in
        headless or non-CUDA environments.

    Inputs:
        run_command (Callable): Subprocess execution callable (injectable for testing).

    Outputs:
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
            (gpu_name, gpu_driver, cuda_version, gpu_info_error).
    """
    gpu_name: Optional[str] = None
    gpu_driver: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_info_error: Optional[str] = None

    try:
        res_driver = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if res_driver.returncode == 0 and res_driver.stdout.strip():
            first_line = res_driver.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in first_line.split(",")]
            if len(parts) >= 2:
                gpu_name, gpu_driver = parts[0], parts[1]
            elif len(parts) == 1:
                gpu_name = parts[0]
        else:
            err = res_driver.stderr.strip() if res_driver.stderr else "Non-zero exit"
            gpu_info_error = f"nvidia-smi query failed: {err}"

        res_cuda = run_command(["nvidia-smi"], capture_output=True, text=True)
        if res_cuda.returncode == 0 and res_cuda.stdout:
            match = re.search(r"CUDA Version:\s*([0-9.]+)", res_cuda.stdout)
            if match:
                cuda_version = match.group(1)
        elif not gpu_info_error:
            err = res_cuda.stderr.strip() if res_cuda.stderr else "Non-zero exit"
            gpu_info_error = f"nvidia-smi CUDA version check failed: {err}"

    except Exception as exc:
        gpu_info_error = str(exc)

    return gpu_name, gpu_driver, cuda_version, gpu_info_error


def verify_engine_model(
    engine: str, manifest_path: Union[str, Path] = "models/manifest.yaml"
) -> None:
    """Verify cryptographic checksums of the model registered for the given engine.

    Purpose:
        Ensure that engine weights are authentic and intact before starting a benchmark run.

    Inputs:
        engine (str): Adapter engine name (e.g. llamacpp_local, vllm_local).
        manifest_path (str | Path): Path to models/manifest.yaml.

    Errors:
        ValueError: If engine is not registered in ENGINE_MODEL_IDS.
        ModelChecksumError: If model is marked unverified or artifact hashes mismatch.
        FileNotFoundError: If model files are missing.
    """
    if engine not in ENGINE_MODEL_IDS:
        raise ValueError(
            f"Unknown engine '{engine}'; valid engines: {sorted(ENGINE_MODEL_IDS.keys())}"
        )

    model_id = ENGINE_MODEL_IDS[engine]
    verify_single_model_files(model_id, manifest_path=manifest_path)


def build_manifest(
    run_id: str,
    started_at: str,
    finished_at: str,
    engine: str,
    quantization: str,
    concurrency: int,
    duration_seconds: int,
    injected_failure: Optional[str],
    git_commit: str,
    git_dirty: bool,
    gpu_name: Optional[str],
    gpu_driver: Optional[str],
    cuda_version: Optional[str],
    gpu_info_error: Optional[str],
    output_files: Dict[str, str],
    manifest_path: Union[str, Path] = "models/manifest.yaml",
    lockfiles: Optional[List[str]] = None,
    config_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble dictionary conforming to schema_version 1 run manifest format.

    Purpose:
        Create complete structured metadata record linking runtime configuration,
        git state, lockfiles, and output telemetry hashes.

    Outputs:
        Dict[str, Any]: Run manifest dictionary containing all REQUIRED_MANIFEST_FIELDS.
    """
    if engine not in ENGINE_MODEL_IDS:
        raise ValueError(f"Unknown engine: {engine}")

    target_lockfiles = lockfiles if lockfiles is not None else DEFAULT_LOCKFILES
    target_config_files = config_files if config_files is not None else DEFAULT_CONFIG_FILES

    lockfile_hashes = hash_files(target_lockfiles)

    config_hashes: Dict[str, str] = {}
    config_files_missing: List[str] = []
    for cfg in target_config_files:
        norm_key = str(cfg).replace("\\", "/")
        if Path(cfg).is_file():
            config_hashes[norm_key] = model_manifest.sha256_of_file(Path(cfg))
        else:
            config_files_missing.append(norm_key)

    manifest_data = model_manifest.load_manifest(manifest_path)
    models = manifest_data.get("models", {})

    target_model_id = ENGINE_MODEL_IDS[engine]
    if target_model_id not in models:
        raise KeyError(f"Engine model '{target_model_id}' not found in manifest")
    m_entry = models[target_model_id]

    engine_model = {
        "id": m_entry["id"],
        "revision": m_entry["revision"],
        "quantization": m_entry["quantization"],
    }
    model_revisions = {mid: str(m["revision"]) for mid, m in models.items()}

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "engine": engine,
        "quantization": quantization,
        "concurrency": concurrency,
        "duration_seconds": duration_seconds,
        "injected_failure": injected_failure,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "lockfile_hashes": lockfile_hashes,
        "config_hashes": config_hashes,
        "config_files_missing": config_files_missing,
        "engine_model": engine_model,
        "model_revisions": model_revisions,
        "routing_rule_set_version": None,
        "slo_version": None,
        "version_notes": (
            "routing rule set versions start in Phase 17; config/slo.yaml starts in Phase 14"
        ),
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "cuda_version": cuda_version,
        "gpu_info_error": gpu_info_error,
        "output_files": output_files,
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], runs_dir: Path = Path("runs")) -> Path:
    """Serialize and write run manifest JSON file to disk.

    Purpose:
        Persist manifest to runs/<run_id>/manifest.json with deterministic formatting.

    Inputs:
        manifest (Dict[str, Any]): Manifest data to serialize.
        runs_dir (Path, optional): Directory to place run subfolders (default: Path('runs')).

    Outputs:
        Path: Path to written manifest.json file.

    Errors:
        FileExistsError: If runs/<run_id>/manifest.json already exists.
    """
    run_id = str(manifest["run_id"])
    target_dir = runs_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "manifest.json"

    if target_file.exists():
        raise FileExistsError(f"Run manifest file already exists: {target_file}")

    content = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    target_file.write_text(content, encoding="utf-8", newline="\n")
    return target_file


def manifest_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of written manifest file."""
    return model_manifest.sha256_of_file(path)


async def store_manifest(
    manifest: Dict[str, Any],
    sha: str,
    dsn: str,
    connect: Any = asyncpg.connect,
) -> None:
    """Insert manifest record into postgres run_manifests table.

    Purpose:
        Ensure durable database storage of run metadata linked by cryptographic hash.

    Inputs:
        manifest (Dict[str, Any]): Manifest dictionary.
        sha (str): Hex SHA-256 digest of written manifest file.
        dsn (str): PostgreSQL connection DSN.
        connect (Callable, optional): Database connector (injectable for unit tests).

    Errors:
        asyncpg.UniqueViolationError / Exception: If run_id already exists or database fails.
    """
    conn = await connect(dsn)
    try:
        created_dt = datetime.fromisoformat(manifest["started_at"])
        query = (
            "INSERT INTO run_manifests ("
            "    run_id, created_at, git_commit, engine, quantization,"
            "    concurrency, manifest_sha256, manifest"
            ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb);"
        )
        await conn.execute(
            query,
            manifest["run_id"],
            created_dt,
            manifest["git_commit"],
            manifest["engine"],
            manifest["quantization"],
            int(manifest["concurrency"]),
            sha,
            json.dumps(manifest),
        )
    finally:
        await conn.close()


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments matching orchestrator interface."""
    parser = argparse.ArgumentParser(
        description="Run benchmark with cryptographic manifest generation and storage."
    )
    parser.add_argument("--engine", default="vllm_local", help="Adapter engine name.")
    parser.add_argument("--quantization", default="none", help="Quantization label.")
    parser.add_argument("--duration", type=int, default=30, help="Seconds per run.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Single concurrency level, or omit for default matrix.",
    )
    parser.add_argument(
        "--injected-failure",
        dest="injected_failure",
        default=None,
        help="Failure scenario name or None.",
    )
    return parser.parse_args()


async def run_benchmark_with_manifest_cli() -> None:
    """Main CLI entrypoint for running benchmark with manifest tracking."""
    args = parse_arguments()

    dsn = os.environ.get("LANDAUER_PG_DSN", "").strip()
    if not dsn:
        sys.stderr.write(
            "Error: LANDAUER_PG_DSN environment variable is missing or empty.\n"
        )
        sys.exit(1)

    # Import orchestrator inside CLI function to avoid module-level sys.exit in test environments
    import orchestrator

    levels = (
        [args.concurrency]
        if args.concurrency is not None
        else orchestrator.DEFAULT_CONCURRENCY_LEVELS
    )

    for concurrency_level in levels:
        verify_engine_model(args.engine)

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()

        print(
            f"[run_with_manifest] Starting run: engine={args.engine} "
            f"quant={args.quantization} concurrency={concurrency_level} "
            f"duration={args.duration}s run_id={run_id}"
        )

        telemetry_path = await orchestrator.run_benchmark(
            engine=args.engine,
            quantization=args.quantization,
            concurrency=concurrency_level,
            duration_seconds=args.duration,
            run_id=run_id,
            injected_failure=args.injected_failure,
        )

        finished_at = datetime.now(timezone.utc).isoformat()

        power_log_path = Path(f"power_draw_{run_id}.log")
        output_files: Dict[str, str] = {}
        if Path(telemetry_path).is_file():
            output_files[str(telemetry_path).replace("\\", "/")] = (
                model_manifest.sha256_of_file(telemetry_path)
            )
        if power_log_path.is_file():
            output_files[str(power_log_path).replace("\\", "/")] = (
                model_manifest.sha256_of_file(power_log_path)
            )

        git_commit, git_dirty = get_git_info()
        gpu_name, gpu_driver, cuda_ver, gpu_err = get_gpu_info()

        manifest = build_manifest(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            engine=args.engine,
            quantization=args.quantization,
            concurrency=concurrency_level,
            duration_seconds=args.duration,
            injected_failure=args.injected_failure,
            git_commit=git_commit,
            git_dirty=git_dirty,
            gpu_name=gpu_name,
            gpu_driver=gpu_driver,
            cuda_version=cuda_ver,
            gpu_info_error=gpu_err,
            output_files=output_files,
        )

        manifest_file = write_manifest(manifest)
        sha = manifest_sha256(manifest_file)
        await store_manifest(manifest, sha, dsn)

        print(f"[run_with_manifest] Stored manifest: {manifest_file} (sha256={sha})")


if __name__ == "__main__":
    asyncio.run(run_benchmark_with_manifest_cli())
