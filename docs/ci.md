# Phase 10 — CI Architecture and Operations

This document describes the Continuous Integration (CI) configuration for Landauer's Limit, covering the test suites, execution jobs, timeout limits, local reproduction steps, and known issues deferred to later phases.

---

## 1. CI Jobs Overview

The CI pipeline runs via GitHub Actions (`.github/workflows/ci.yml`) on `ubuntu-latest` and consists of two parallel jobs:

### Job 1: `unit-and-integration`
- **Linting**: Static syntax analysis with `make lint` (`ruff check --select F,E9`).
- **Type Checking**: Strict type validation with `make typecheck` (`mypy --ignore-missing-imports --follow-imports=silent`) on Phase 9 production modules and CI tooling.
- **ONNX Export & Caching**: Restores cached `all-MiniLM-L6-v2.onnx` and Hugging Face tokenizer cache, or generates the ONNX model using CPU PyTorch.
- **Unit and Integration Tests**: Executes `make test-ci` (`pytest -m "not gpu"`) over the CPU-safe Phase 1–9 test suite.

### Job 2: `stack-smoke`
- **Credential Generation**: Dynamically provisions random secrets for Postgres, Grafana, and Gateway API keys using `secrets.token_urlsafe`, masking values with `::add-mask::`.
- **Model Acquisition**: Restores or downloads `qwen2.5-0.5b-instruct-q4_k_m.gguf` from Hugging Face and verifies its SHA256 checksum.
- **Docker Compose Stack**: Builds images and starts Postgres, Prometheus, Grafana, llama.cpp, and Gateway with healthcheck barriers.
- **Smoke Validation**: Runs `scripts/ci_smoke.py` validating authenticated SSE token streaming from the gateway and Grafana database health.
- **Cleanup**: Purges containers, networks, and named/anonymous volumes via `docker compose down --volumes`.

---

## 2. Test Suites: `test-ci` vs `test-gpu`

The test runner separates fast, deterministic CPU checks from hardware-dependent GPU checks:

- **`make test-ci`**:
  Runs `pytest -m "not gpu"` against deterministic CPU-only unit and integration tests. No CUDA hardware or live model engines are required.
- **`make test-gpu`**:
  Runs `pytest -m gpu` against GPU-dependent test scripts (`tests/test_gpu_scripts.py`, `test_live_streaming.py`). Requires local CUDA hardware, NVIDIA drivers, and live inference endpoints.

### "Deselected ≠ Skipped"
Tests filtered out in CI with `-m "not gpu"` are **deselected**, not skipped. Deselection occurs at collection time based on test markers. Both suites are mandatory for overall project verification: CI validates CPU components on pull requests, while local hardware testing validates GPU components. Per Phase 30 criteria, production sign-off requires both suites passing with zero skipped tests.

---

## 3. CI Test File List

The single source of truth for CI test files is defined in `Makefile` under `CI_TEST_FILES`. Printed with `make print-ci-files`:

```
tests/test_ci_scripts.py tests/test_phase6_prefix_invalidation.py tests/test_phase2_payload.py tests/test_phase3_guardrail.py test_phase4.py tests/test_phase7_failure_injection.py tests/test_config_startup.py tests/test_phase9_auth.py tests/test_phase9_engine_config.py tests/test_phase9_gpu_monitor.py tests/test_phase9_monitoring.py tests/test_phase9_pg.py tests/test_phase9_telemetry.py tests/test_generate_report.py
```

---

## 4. Startup Timeout LIMIT Derivation

Per project Ground Rule 2 (every magic number must be derived and documented), the `--wait-timeout` flag for `docker compose up --wait` is derived from empirical timing measurements on local WSL2 Docker hardware.

### Timed Runs:
- **Run 1**: 75.45 s
- **Run 2**: 92.70 s
- **Run 3**: 50.50 s

### Derivation:
- **Slowest run**: 92.70 s
- **Safety factor**: 3×
- **Calculation**: 3 × 92.70 s = 278.11 s
- **LIMIT**: Rounded up to next 10 s = **280 s** (`--wait-timeout 280`).

---

## 5. Local Reproduction of Job 2 (`stack-smoke`)

To reproduce the full containerized smoke test locally:

1. **Download the GGUF model**:
   ```bash
   mkdir -p ci-models
   curl -sSL "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
     -o ci-models/qwen2.5-0.5b-instruct-q4_k_m.gguf
   ```
2. **Verify SHA256 checksum**:
   ```bash
   echo "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db  ci-models/qwen2.5-0.5b-instruct-q4_k_m.gguf" | sha256sum -c -
   ```
3. **Configure environment and tenants**:
   Ensure `.env` contains valid credentials and generate `deploy/ci-gateway-config/tenants.yaml` with the SHA256 hash of `LANDAUER_API_KEY`.
4. **Prepare telemetry directory**:
   ```bash
   mkdir -p telemetry && chmod 777 telemetry
   ```
5. **Build and start stack**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.ci.yml build
   docker compose -f docker-compose.yml -f docker-compose.ci.yml up --wait
   ```
6. **Execute smoke validation**:
   ```bash
   python scripts/ci_smoke.py
   ```
7. **Tear down stack**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.ci.yml down --volumes
   ```

---

## 6. Known Issues (Deferred from Phase 10)

1. **`ground_truth.jsonl` Duplicate Record**:
   The committed `ground_truth.jsonl` contains 7 records where `guardrail_timeout` is duplicated. Ground truth is frozen in Phase 10. Deduplication is scheduled for Phase 29, ensuring the 6 unique run IDs continue to match telemetry event records.
2. **`guardrail.GUARDRAIL_TIMEOUT` Derivation**:
   `guardrail.GUARDRAIL_TIMEOUT` is currently set to `10.0` (a development value, violating Ground Rule 2). The empirical derivation measured during Phase 3 benchmarking is:
   $$\text{Derivation} = 65.32\text{ ms} \times 1.5 = 97.98\text{ ms}$$
   Updating this constant is deferred until Phase 11.
3. failure_injection.py (frozen Phase 7) cannot pass standalone on current main: its httpx requests carry no X-API-Key (401 since Phase 9 auth), and its guardrail_timeout scenario sleeps 0.2 s against GUARDRAIL_TIMEOUT = 10.0. tests/test_phase7_failure_injection.py works around both without editing the script (injects the key into httpx.AsyncClient.post, sets GUARDRAIL_TIMEOUT to 0.05, clears circuit_breaker._breakers between tests). Phase 29 must account for this before relying on the script.
4. The Phase 1 red-run proof added a 201st record (count check failed), not a duplicate ID.
