"""Configuration loading for the data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or repository_root() / "configs" / "data.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return value


def configured_data_root(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    configured = config.get("paths", {}).get("data_root", "data")
    return Path(str(configured))
