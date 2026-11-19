"""
tests/test_phase12_pricing.py
=============================
Unit tests for Phase 12 electricity pricing and engine energy cost calculation:
- Committed config/pricing.yaml validation and loading.
- Hand-worked mathematical verification of energy_cost formula.
- Edge case handling: unknown engine, negative duration, rate 0, missing URL.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pricing import (
    default_pricing_path,
    energy_cost,
    load_pricing,
)


def test_committed_pricing_yaml_loads() -> None:
    """Verifies that the committed config/pricing.yaml exists, loads, and validates."""
    pricing_path = default_pricing_path()
    assert pricing_path.is_file(), f"Expected pricing file at {pricing_path}"

    config = load_pricing(pricing_path)
    assert config.currency == "INR"
    assert config.electricity_rate_per_kwh == 7.20
    assert config.rate_source.supplier == "Kerala State Electricity Board Ltd (KSEB)"
    assert config.rate_source.url.startswith("https://kseb.in")
    assert "vllm_local" in config.engine_power_watts
    assert "llamacpp_local" in config.engine_power_watts
    assert "custom_runtime" in config.engine_power_watts
    assert "colab_cloud" in config.engine_power_watts
    assert config.engine_power_watts["vllm_local"] == 67.95
    assert config.engine_power_watts["llamacpp_local"] == 69.82
    assert config.engine_power_watts["custom_runtime"] == 31.54
    assert config.engine_power_watts["colab_cloud"] == 5.29


def test_energy_cost_hand_worked_case(tmp_path: Path) -> None:
    """
    Verifies calculation with hand-worked numbers:
    1000 W for 3600 s at 7.20 INR/kWh:
    kWh = (1000 W * 3600 s) / 3,600,000 = 1.0 kWh
    Cost = 1.0 kWh * 7.20 = 7.20 INR.
    """
    sample_yaml = tmp_path / "sample_pricing.yaml"
    sample_yaml.write_text(
        yaml.safe_dump(
            {
                "currency": "INR",
                "electricity_rate_per_kwh": 7.20,
                "rate_source": {
                    "supplier": "Test Supplier",
                    "tariff": "Test Tariff",
                    "url": "https://example.com/tariff",
                    "effective_from": "2025-01-01",
                    "valid_until": "2027-01-01",
                    "retrieved_on": "2026-11-19",
                },
                "engine_power_watts": {
                    "benchmark_engine": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_pricing(sample_yaml)
    cost = energy_cost(config, "benchmark_engine", duration_seconds=3600.0)
    assert pytest.approx(cost, abs=1e-6) == 7.20


def test_energy_cost_unknown_engine_raises() -> None:
    """Verifies that an unknown engine name raises ValueError naming the engine."""
    config = load_pricing(default_pricing_path())
    with pytest.raises(ValueError, match="Unknown engine 'nonexistent_engine'"):
        energy_cost(config, "nonexistent_engine", duration_seconds=60.0)


def test_energy_cost_negative_duration_raises() -> None:
    """Verifies that negative duration_seconds raises ValueError."""
    config = load_pricing(default_pricing_path())
    with pytest.raises(ValueError, match="duration_seconds cannot be negative"):
        energy_cost(config, "vllm_local", duration_seconds=-5.0)


def test_load_pricing_rate_zero_raises(tmp_path: Path) -> None:
    """Verifies that an electricity rate of 0 (or negative) fails validation with ValueError."""
    zero_rate_yaml = tmp_path / "zero_rate.yaml"
    zero_rate_yaml.write_text(
        yaml.safe_dump(
            {
                "currency": "INR",
                "electricity_rate_per_kwh": 0.0,
                "rate_source": {
                    "supplier": "Test",
                    "tariff": "Test",
                    "url": "https://example.com",
                    "effective_from": "2025-01-01",
                    "valid_until": "2027-01-01",
                    "retrieved_on": "2026-11-19",
                },
                "engine_power_watts": {"vllm_local": 67.95},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is invalid"):
        load_pricing(zero_rate_yaml)


def test_load_pricing_missing_url_raises(tmp_path: Path) -> None:
    """Verifies that a missing required rate_source field (e.g. url) raises ValueError."""
    missing_url_yaml = tmp_path / "missing_url.yaml"
    missing_url_yaml.write_text(
        yaml.safe_dump(
            {
                "currency": "INR",
                "electricity_rate_per_kwh": 7.20,
                "rate_source": {
                    "supplier": "Test",
                    "tariff": "Test",
                    "effective_from": "2025-01-01",
                    "valid_until": "2027-01-01",
                    "retrieved_on": "2026-11-19",
                },
                "engine_power_watts": {"vllm_local": 67.95},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is invalid"):
        load_pricing(missing_url_yaml)


def test_default_pricing_path_honors_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that LANDAUER_PRICING_PATH environment variable overrides default path."""
    custom_path = tmp_path / "custom_pricing.yaml"
    monkeypatch.setenv("LANDAUER_PRICING_PATH", str(custom_path))
    assert default_pricing_path() == custom_path
