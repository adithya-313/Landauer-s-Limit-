"""
pricing.py
==========
Pricing configuration loader and energy cost calculation for inference engines.
Reads electricity rates and estimated engine wattage from config/pricing.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml


class RateSource(BaseModel):
    """
    Source metadata documenting electricity tariff origin and validity.
    """

    supplier: str = Field(..., min_length=1)
    tariff: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    effective_from: str = Field(..., min_length=1)
    valid_until: str = Field(..., min_length=1)
    retrieved_on: str = Field(..., min_length=1)


class PricingConfig(BaseModel):
    """
    Data model representing global electricity pricing and engine power draw estimates.
    """

    currency: str = Field(..., pattern=r"^[A-Z]{3}$")
    electricity_rate_per_kwh: float = Field(..., gt=0.0)
    rate_source: RateSource
    engine_power_watts: dict[str, float]

    @field_validator("engine_power_watts")
    @classmethod
    def validate_engine_power_watts(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensures engine power dictionary is non-empty and every watt value is strictly positive."""
        if not v:
            raise ValueError("engine_power_watts must not be empty")
        for engine, watts in v.items():
            if watts <= 0:
                raise ValueError(f"engine '{engine}' power watts must be > 0, got {watts}")
        return v


def default_pricing_path() -> Path:
    """
    Returns the filesystem path to the pricing configuration file.
    Honors LANDAUER_PRICING_PATH if set, otherwise defaults to <repo root>/config/pricing.yaml.
    """
    env_path = os.getenv("LANDAUER_PRICING_PATH")
    if env_path is not None and env_path.strip():
        return Path(env_path)
    repo_root = Path(__file__).resolve().parent
    return repo_root / "config" / "pricing.yaml"


def load_pricing(path: str | Path) -> PricingConfig:
    """
    Loads and validates pricing configuration from a YAML file.

    Inputs:
    - path: Filesystem path to the pricing YAML file.

    Returns:
    - PricingConfig model containing validated pricing settings.

    Errors:
    - Raises ValueError naming the file and problem if file is missing,
      unparseable, structurally invalid, or fails validation.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Pricing configuration file '{p}' does not exist")

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse pricing configuration file '{p}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Pricing configuration file '{p}' must contain a top-level mapping")

    try:
        return PricingConfig(**data)
    except ValidationError as exc:
        raise ValueError(f"Pricing configuration file '{p}' is invalid: {exc}") from exc


def energy_cost(pricing: PricingConfig, engine: str, duration_seconds: float) -> float:
    """
    Calculates the energy cost in pricing.currency for running engine for duration_seconds.

    Formula:
    kwh = watts * duration_seconds / 3_600_000
    cost = kwh * rate

    Inputs:
    - pricing: Validated PricingConfig.
    - engine: Target engine name.
    - duration_seconds: Run duration in seconds (must be non-negative).

    Returns:
    - float cost in configured currency.

    Errors:
    - Raises ValueError if duration_seconds < 0.
    - Raises ValueError naming the unknown engine if engine is not in pricing.engine_power_watts.
    """
    if duration_seconds < 0:
        raise ValueError(f"duration_seconds cannot be negative, got {duration_seconds}")

    if engine not in pricing.engine_power_watts:
        raise ValueError(f"Unknown engine '{engine}' not found in pricing.engine_power_watts")

    watts = pricing.engine_power_watts[engine]
    kwh = (watts * duration_seconds) / 3_600_000.0
    return float(kwh * pricing.electricity_rate_per_kwh)
