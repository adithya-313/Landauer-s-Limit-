# Reproducibility Guide

This document describes the reproducibility architecture for Landauer's Limit,
including dependency pin locks, container image digests, model artifact checksums,
run manifests, Alembic schema migrations, and full report reproduction.

---

## 1. Locked Dependencies

To prevent upstream supply-chain variations or unpinned library breaks, all execution
environments use exact lockfiles compiled with `--generate-hashes`.

- **`requirements-gateway.lock`**: Installed inside the gateway container. Generated on
  Linux (`x86_64-unknown-linux-gnu`) targeting Python 3.12.
- **`requirements-ci.lock`**: Installed during Continuous Integration test runs on Linux.
- **`requirements-ci-windows.lock`**: Installed for local Windows development venvs.
- **`requirements-migrations.lock`**: Installed for running Alembic database migrations and
  schema parity checks across all environments.

### Compilation Commands

```bash
uv pip compile requirements-gateway.txt --python-platform x86_64-unknown-linux-gnu --python-version 3.12 --generate-hashes -o requirements-gateway.lock
uv pip compile requirements-ci.txt --python-platform x86_64-unknown-linux-gnu --python-version 3.12 --generate-hashes -o requirements-ci.lock
uv pip compile requirements-ci.txt --python-platform x86_64-pc-windows-msvc --python-version 3.12 --generate-hashes -o requirements-ci-windows.lock
uv pip compile requirements-migrations.txt --generate-hashes -o requirements-migrations.lock
```

---

## 2. Pinned Container Image Digests

All base images in `docker-compose.yml`, `docker-compose.ci.yml`, and `Dockerfile` are pinned by
immutable SHA-256 digest:

- **PostgreSQL 15**: `postgres:15@sha256:5a407822929c3789ff5154d3d35bd7e2090318149344c8a3d4ca942b035822ed`
- **Prometheus**: `prom/prometheus:v2.54.1@sha256:6eec08bf09a8039e1e9ab8da82db2734e9cf74c052820658fe96ee243bc5d96a`
- **Grafana**: `grafana/grafana:11.1.0@sha256:4fb0be712be1d3b0c51cb9c3be79f15dbb9cfb34a6efbc52bc5a6fef95bdf561`
- **Python 3.12 Slim**: `python:3.12-slim@sha256:5424ecf2a58d6ebef427da97ce1d114f6b15822830fccbfdffdd29ff8e6371cb`

---

## 3. Model Manifest & Verification

Model weights, tokenizer configurations, and exported ONNX artifacts are tracked in
`models/manifest.yaml`. Cryptographic hashes ensure that identical model weights are loaded
across runs.

- **Integrity Verification**: `python scripts/verify_models.py` verifies all registered models
  against local files or remote Hugging Face metadata.
- **Guardrail ONNX Sidecar**: The safety classifier ONNX model is verified using a cryptographic
  SHA-256 sidecar file (`guardrail_classifier.onnx.sha256`).

---

## 4. Run Manifests

Every benchmark invocation using `run_with_manifest.py` generates an immutable JSON metadata
record under `runs/<run_id>/manifest.json` and inserts a matching row into the `run_manifests`
PostgreSQL database table.

Each run manifest captures:
- Exact Git commit SHA and dirty state.
- SHA-256 digests of all active dependency lockfiles.
- SHA-256 digests of runtime configurations (`config.yaml`, `tenants.yaml`).
- Model identifier, revision, and artifact hashes.
- NVIDIA GPU driver and CUDA version (queried dynamically via `nvidia-smi`).
- Execution duration, concurrency, quantization, and injected failure parameters.
- SHA-256 digests of all output telemetry (`telemetry_<run_id>.jsonl`) and power logs.

Manifests are independently validated using:
```bash
python scripts/verify_run_manifest.py runs/<run_id>/manifest.json
```

---

## 5. Database Schema & Alembic Migrations

Database schema versioning is managed via Alembic under `migrations/`.

- **Migration baseline**: `0001_baseline_phase_9_schema.py` initializes the Phase 9 tables (`telemetry_events`, `tenants`).
- **Run manifests table**: `0002_add_run_manifests_table.py` creates `run_manifests`.
- **Parity verification**: `scripts/check_schema_parity.py` ensures that a fresh database initialized
  from `db/init/001_schema.sql` matches the schema produced by running `alembic upgrade head`
  from scratch. CI runs this check on every pull request.

---

## 6. Report Reproduction

The Stage 8 research findings and report are reproduced byte-for-byte from raw telemetry:

```bash
make reproduce-report
```

This target runs `scripts/reproduce_report.py`, which:
1. Validates cryptographic hashes of all 24 raw telemetry files in `archive/phase8/SHA256SUMS`.
2. Generates `report.md` from the archived artifacts.
3. Compares the newly generated report against the committed `report.md` to guarantee
   byte-for-byte identity.

---

## 7. Reference Tables

<!-- generated:start -->
### Run Manifest Fields

| Field Name | Type | Description |
| --- | --- | --- |
| `schema_version` | `int` | Schema specification version (currently 1). |
| `run_id` | `str (UUID)` | Unique execution UUID identifying telemetry and logs. |
| `started_at` | `str (ISO-8601 UTC)` | Benchmark execution initiation timestamp. |
| `finished_at` | `str (ISO-8601 UTC)` | Benchmark completion timestamp. |
| `engine` | `str` | Inference engine key (e.g. llamacpp_local, vllm_local). |
| `quantization` | `str` | Quantization format specification (e.g. q4, awq). |
| `concurrency` | `int` | Number of concurrent simulated client requests. |
| `duration_seconds` | `int` | Target benchmark duration in seconds. |
| `injected_failure` | `str | None` | Active failure injection scenario name, if any. |
| `git_commit` | `str` | Exact 40-character Git SHA at time of benchmark run. |
| `git_dirty` | `bool` | Flag indicating whether uncommitted git changes existed. |
| `lockfile_hashes` | `dict[str, str]` | SHA-256 digests of all dependency lockfiles. |
| `config_hashes` | `dict[str, str]` | SHA-256 digests of runtime config and tenant files. |
| `config_files_missing` | `list[str]` | List of optional configuration files that were missing. |
| `engine_model` | `dict` | Model metadata entry from models/manifest.yaml for the active engine. |
| `model_revisions` | `dict[str, str]` | Pinned model repository revisions (git commit SHAs). |
| `routing_rule_set_version` | `str` | Active routing rule set version string from config.yaml. |
| `slo_version` | `str` | Active service level objective policy version from config.yaml. |
| `version_notes` | `str` | Reproducibility notes and assumptions recorded for this run. |
| `gpu_name` | `str | None` | GPU hardware name reported by nvidia-smi. |
| `gpu_driver` | `str | None` | GPU driver version string reported by nvidia-smi. |
| `cuda_version` | `str | None` | CUDA runtime / driver version from nvidia-smi. |
| `gpu_info_error` | `str | None` | Error diagnostics if nvidia-smi query failed. |
| `output_files` | `dict[str, str]` | SHA-256 hashes of telemetry and power logs. |

### Registered Models

| Model ID | Source | Quantization | Verified | Note |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | `huggingface` | `Q4_K_M` | Yes | verified by CI step 6 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `huggingface` | `fp16` | Yes | Source fp16 weights and tokenizer from Hugging Face Hub |
| `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | `huggingface` | `awq-int4` | No | The hashes came from the Hugging Face API, not local files, because the model only ran inside Docker. |
| `custom_runtime` | `huggingface` | `bnb-4bit-on-load` | Yes | The 4-bit weights are produced in memory so the source files are what get checksummed. |
| `ollama/qwen2.5:1.5b` | `ollama` | `Q4_K_M` | Yes | Ollama pulled image manifest and layer blobs from registry.ollama.ai |
| `sentence-transformers/all-MiniLM-L6-v2` | `local_export` | `fp32` | Yes | The exported .onnx is verified against its sidecar. |

### Primary Makefile Targets

| Target | Description |
| --- | --- |
| `make all` | Run lint, typecheck, test-ci, and reproduce-report targets in sequence. |
| `make lint` | Run ruff code checks against all tracked repository modules. |
| `make print-ci-files` | Print lists of files targeted for lint, typecheck, and CI tests. |
| `make reproduce-report` | Verify phase8 archive checksums and reproduce report.md. |
| `make test-ci` | Execute full continuous integration pytest suite. |
| `make test-gpu` | Execute GPU-dependent benchmark and adapter test suites. |
| `make typecheck` | Run mypy static type checking against core repository modules. |
<!-- generated:end -->

