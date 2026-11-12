from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ENGINE_ID_REGEX = re.compile(r"^[a-z0-9_]{1,64}$")
SUPPORTED_ENGINE_TYPES = ("vllm", "llamacpp", "colab", "custom_runtime")


class EngineSpec(BaseModel):
    """
    Specification for a single execution engine adapter in the platform.

    Fields:
    - id: Unique engine identifier, matching ^[a-z0-9_]{1,64}$.
    - type: Engine adapter type (vllm, llamacpp, colab, custom_runtime).
    - url: Optional network URL; required for network engines, forbidden for custom_runtime.
    - tier: Human-readable metadata tier label (e.g. premium, free, experimental).
    """

    id: str = Field(..., pattern=r"^[a-z0-9_]{1,64}$")
    type: Literal["vllm", "llamacpp", "colab", "custom_runtime"]
    url: Optional[str] = None
    tier: str

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: Any) -> str:
        """Ensures tier is a non-empty string and strips surrounding whitespace."""
        if not isinstance(v, str):
            raise ValueError("tier must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("tier must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def validate_engine_url(self) -> EngineSpec:
        """
        Enforces engine URL constraints based on engine type.
        Network-based engines (vllm, llamacpp, colab) require a valid http/https URL.
        Embedded engines (custom_runtime) must not have a URL configured.
        """
        if self.type in ("vllm", "llamacpp", "colab"):
            if not self.url:
                raise ValueError(f"url is required for engine type '{self.type}'")
            if not (self.url.startswith("http://") or self.url.startswith("https://")):
                raise ValueError(
                    f"url must start with http:// or https:// for engine type '{self.type}', got '{self.url}'"
                )
        elif self.type == "custom_runtime":
            if self.url is not None:
                raise ValueError(
                    f"url must be None/absent for engine type 'custom_runtime', got '{self.url}'"
                )
        return self


def default_config_path() -> Path:
    """
    Resolves the configuration file path.

    Checks the LANDAUER_CONFIG_PATH environment variable first.
    If unset or empty, defaults to 'config.yaml' located in the repository root.

    Returns:
    - Path object pointing to the resolved config file location.
    """
    repo_root = Path(__file__).resolve().parent
    env_path = os.environ.get("LANDAUER_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            return repo_root / p
        return p
    return repo_root / "config.yaml"


def load_engine_specs(path: str | Path) -> list[EngineSpec]:
    """
    Reads and validates engine specifications from a YAML configuration file.

    Inputs:
    - path: File system path to the YAML configuration file.

    Returns:
    - List of validated EngineSpec models in file order.

    Side Effects / Errors:
    - Raises ValueError with clear file and reason details if file is missing,
      not a mapping, missing the 'engines' list, contains invalid entries,
      or contains duplicate engine identifiers.
    """
    config_file = Path(path)
    if not config_file.is_file():
        raise ValueError(f"Configuration file not found: '{config_file}'")

    try:
        content = config_file.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except Exception as exc:
        raise ValueError(f"Failed to read or parse YAML in '{config_file}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level structure in '{config_file}' must be a mapping/dict, got {type(data).__name__}"
        )

    if "engines" not in data:
        raise ValueError(f"Missing required 'engines' key in configuration file '{config_file}'")

    engines_data = data["engines"]
    if not isinstance(engines_data, list):
        raise ValueError(
            f"'engines' in '{config_file}' must be a list, got {type(engines_data).__name__}"
        )

    specs: list[EngineSpec] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(engines_data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid engine entry at index {idx} in '{config_file}': expected mapping, got {type(entry).__name__}"
            )
        try:
            spec = EngineSpec.model_validate(entry)
        except Exception as exc:
            raise ValueError(
                f"Invalid engine entry at index {idx} in '{config_file}': {exc}"
            ) from exc

        if spec.id in seen_ids:
            raise ValueError(
                f"Duplicate engine id '{spec.id}' found in configuration file '{config_file}'"
            )
        seen_ids.add(spec.id)
        specs.append(spec)

    return specs


def save_engine_specs(path: str | Path, specs: list[EngineSpec]) -> None:
    """
    Atomically writes engine specifications to a YAML configuration file.

    Writes the payload to a temporary file in the destination directory,
    then replaces the target path atomically via os.replace().
    Comments in config.yaml are not preserved.

    Inputs:
    - path: Target path for the YAML configuration file.
    - specs: List of EngineSpec models to persist.

    Side Effects:
    - Writes to filesystem; overwrites existing file at path atomically.
    - Cleans up temporary files in case of write failure.
    """
    target_path = Path(path).resolve()
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "engines": [
            spec.model_dump(mode="python", exclude_none=True)
            for spec in specs
        ]
    }

    temp_fd, temp_name = tempfile.mkstemp(
        prefix="engine_config_",
        suffix=".tmp",
        dir=str(target_dir),
        text=True,
    )
    temp_file = os.fdopen(temp_fd, "w", encoding="utf-8")
    try:
        yaml.safe_dump(payload, temp_file, sort_keys=False)
        temp_file.flush()
        temp_file.close()
        os.replace(temp_name, target_path)
    finally:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                pass


def build_adapter(spec: EngineSpec) -> Any:
    """
    Constructs and returns the concrete adapter instance for an engine specification.

    Imports adapter classes lazily inside this function to ensure importing
    engine_config remains lightweight without loading heavy dependencies.

    Inputs:
    - spec: Validated EngineSpec defining the adapter type and parameters.

    Returns:
    - Instantiated adapter instance.

    Errors:
    - Raises ValueError if the engine type is unknown or invalid.
    """
    if spec.type == "vllm":
        from adapters.vllm_adapter import VLLMAdapter

        if not spec.url:
            raise ValueError(f"vllm engine '{spec.id}' requires a valid url")
        return VLLMAdapter(base_url=spec.url)

    if spec.type == "llamacpp":
        from adapters.llamacpp_adapter import LlamaCppAdapter

        if not spec.url:
            raise ValueError(f"llamacpp engine '{spec.id}' requires a valid url")
        return LlamaCppAdapter(base_url=spec.url)

    if spec.type == "colab":
        from adapters.colab_adapter import ColabAdapter

        if not spec.url:
            raise ValueError(f"colab engine '{spec.id}' requires a valid url")
        return ColabAdapter(tunnel_url=spec.url)

    if spec.type == "custom_runtime":
        from adapters.custom_runtime_adapter import CustomRuntimeAdapter

        return CustomRuntimeAdapter()

    raise ValueError(f"Unsupported engine type: '{spec.type}'")
