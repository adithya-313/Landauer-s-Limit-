"""
Tests for Phase 7 failure injection and ground truth record generation.

Verifies that each failure scenario triggers its expected protective action and writes
a compliant record to ground_truth.jsonl without mutating production artifacts.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import uuid

import httpx
import pytest

import circuit_breaker
import failure_injection
from failure_injection import FAILURE_SCENARIOS, inject_failure
import protective_actions


@pytest.fixture(autouse=True)
def isolate_action_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolates actions_taken.jsonl and ground_truth.jsonl to temporary paths, resets circuit breakers, and injects API key."""
    temp_ground_truth = tmp_path / "ground_truth.jsonl"
    temp_actions_log = tmp_path / "actions_taken.jsonl"

    monkeypatch.setattr(failure_injection, "GROUND_TRUTH_PATH", temp_ground_truth)
    monkeypatch.setattr(failure_injection, "ACTIONS_LOG", temp_actions_log)
    monkeypatch.setattr(protective_actions, "ACTIONS_LOG", temp_actions_log)

    # Reset circuit breakers to clear any tripped state from previous tests
    circuit_breaker._breakers.clear()

    # Inject API key into httpx calls made by failure injection
    orig_post = httpx.AsyncClient.post

    async def authed_post(self, url, *args, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        if "X-API-Key" not in headers and "Authorization" not in headers:
            headers["X-API-Key"] = os.environ.get("LANDAUER_API_KEY", "")
        kwargs["headers"] = headers
        return await orig_post(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", authed_post)

    yield

    circuit_breaker._breakers.clear()


@pytest.mark.parametrize("scenario_name", FAILURE_SCENARIOS)
def test_single_scenario_injection(scenario_name: str) -> None:
    """Verifies that injecting a single failure scenario records ground truth for that run."""
    run_id = f"test-run-{uuid.uuid4().hex[:8]}"

    # NOTE: Telemetry record correlation ('every run ID matches a telemetry record')
    # is verified in full stack/pipeline integration tests, not in this unit test.
    asyncio.run(inject_failure(scenario_name, run_id))

    ground_truth_file = failure_injection.GROUND_TRUTH_PATH
    assert ground_truth_file.is_file(), f"Ground truth file not written for {scenario_name}"

    records = [
        json.loads(line)
        for line in ground_truth_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["run_id"] == run_id
    assert records[0]["failure_type"] == scenario_name


def test_all_scenarios_aggregate_injection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies running all six failure scenarios aggregates exactly 6 complete ground truth records."""
    agg_ground_truth = tmp_path / "agg_ground_truth.jsonl"
    agg_actions = tmp_path / "agg_actions.jsonl"
    monkeypatch.setattr(failure_injection, "GROUND_TRUTH_PATH", agg_ground_truth)
    monkeypatch.setattr(failure_injection, "ACTIONS_LOG", agg_actions)
    monkeypatch.setattr(protective_actions, "ACTIONS_LOG", agg_actions)

    for scenario_name in FAILURE_SCENARIOS:
        run_id = f"agg-run-{uuid.uuid4().hex[:8]}"
        asyncio.run(inject_failure(scenario_name, run_id))

    assert agg_ground_truth.is_file()
    records = [
        json.loads(line)
        for line in agg_ground_truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 6, f"Expected 6 records, got {len(records)}"

    recorded_types = [r["failure_type"] for r in records]
    assert set(recorded_types) == set(FAILURE_SCENARIOS)
    assert len(recorded_types) == len(set(recorded_types))  # Each appears exactly once

    required_keys = {"run_id", "failure_type", "root_cause", "remediation", "timestamp"}
    for record in records:
        missing = required_keys - set(record.keys())
        assert not missing, f"Record {record} is missing required fields: {missing}"
