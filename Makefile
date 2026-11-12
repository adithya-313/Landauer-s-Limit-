# Makefile - Single source of truth for CI linting, typechecking, and test execution.

PYTHON ?= python3
PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy

# Modules implemented in Phase 9 to be strictly type-checked
TYPECHECK_MODULES := \
	engine_config.py \
	tenants.py \
	telemetry.py \
	pg_writer.py \
	admin_routes.py \
	gpu_monitor.py \
	payload_guard.py \
	scripts/ci_smoke.py

# CI test files executed in headless, CPU-only environments
CI_TEST_FILES := \
	tests/test_ci_scripts.py \
	tests/test_phase6_prefix_invalidation.py \
	tests/test_phase2_payload.py \
	tests/test_phase3_guardrail.py \
	test_phase4.py \
	tests/test_phase7_failure_injection.py \
	tests/test_config_startup.py \
	tests/test_phase9_auth.py \
	tests/test_phase9_engine_config.py \
	tests/test_phase9_gpu_monitor.py \
	tests/test_phase9_monitoring.py \
	tests/test_phase9_pg.py \
	tests/test_phase9_telemetry.py \
	tests/test_generate_report.py

# GPU and live-server test files executed locally on GPU-enabled infrastructure
GPU_TEST_FILES := \
	tests/test_gpu_scripts.py \
	test_live_streaming.py

# Files covered by static syntax and import checks
LINT_FILES := \
	$(TYPECHECK_MODULES) \
	$(CI_TEST_FILES) \
	$(GPU_TEST_FILES) \
	conftest.py \
	validate.py \
	tests/test_phase6d_1_hashing.py \
	tests/test_phase6d_2_blocks.py \
	tests/test_phase6d_3_registry.py

.PHONY: all lint typecheck test-ci test-gpu print-ci-files

all: lint typecheck test-ci

# Static syntax and error analysis
lint:
	$(RUFF) check --select F,E9 $(LINT_FILES)

# Type validation on Phase 9 production modules
typecheck:
	$(MYPY) --ignore-missing-imports --follow-imports=silent $(TYPECHECK_MODULES)

# CPU unit and integration test suite for CI
# NOTE: Deselecting with -m "not gpu" is not skipping; GPU tests are excluded in CI
# and must be green locally (Phase 30 requires both sets green with zero skips).
test-ci:
	$(PYTEST) -m "not gpu" $(CI_TEST_FILES)

# GPU and live-model-server test suite (run locally with CUDA hardware)
test-gpu:
	$(PYTEST) -m gpu $(GPU_TEST_FILES)

print-ci-files:
	@echo $(CI_TEST_FILES)
