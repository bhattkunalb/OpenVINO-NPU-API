"""Pydantic model registry: load and validate model entries from YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

log = logging.getLogger(__name__)


class ModelEntry(BaseModel):
    """Single model entry in the registry."""
    name: str
    path: str
    task: Literal["chat", "completion", "embedding", "vision"]
    input_type: Literal["text", "image", "tensor"] = "text"
    device: str = "NPU"
    preprocess_fn: str = "default_genai"
    postprocess_fn: str = "default_genai"
    max_tokens: int = 2048
    context_length: int = 4096
    image_size: int | None = None
    lora_adapters: list[str] | None = None

    @field_validator("device")
    @classmethod
    def _uppercase_device(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _validate_custom_fns(self) -> "ModelEntry":
        for attr in ("preprocess_fn", "postprocess_fn"):
            val = getattr(self, attr)
            if val.startswith("custom:") and "." not in val.split(":", 1)[1]:
                raise ValueError(f"[{self.name}] {attr}='{val}' must be 'custom:<module.fn>'")
        return self


class RegistryConfig(BaseModel):
    """Top-level config parsed from models.yaml."""
    models: list[ModelEntry]

    @model_validator(mode="after")
    def _unique_names(self) -> "RegistryConfig":
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate model names: {[n for n in names if names.count(n) > 1]}")
        return self


def load_registry(config_path: str | Path) -> RegistryConfig:
    """Parse and validate models.yaml into a RegistryConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = RegistryConfig.model_validate(raw)
    log.info("Registry: %d model(s) loaded.", len(cfg.models))
    return cfg
