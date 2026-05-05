"""
Pydantic model registry: load, validate, and resolve model entries from YAML.

Field names match the spec exactly:
  task, device, preprocess_fn, postprocess_fn, context_length, max_tokens
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

log = logging.getLogger(__name__)

Task = Literal["chat", "completion", "embedding", "vision"]
PreprocessFn = Literal["default_genai"]  # or "custom:<module.function>"
PostprocessFn = Literal["default_genai"]  # or "custom:<module.function>"


class ModelEntry(BaseModel):
    """
    Single model entry in the registry.
    Fully self-describing: path, task, device, limits, and pipeline hooks.
    """

    name: str
    path: str                               # path to OV IR directory or HF model id
    task: Task
    input_type: Literal["text", "image", "tensor"] = "text"
    device: str = "NPU"
    preprocess_fn: str = "default_genai"    # "default_genai" | "custom:<module.fn>"
    postprocess_fn: str = "default_genai"   # "default_genai" | "custom:<module.fn>"
    max_tokens: int = 2048
    context_length: int = 4096
    # Optional fields
    image_size: Optional[int] = None
    lora_adapters: Optional[list[str]] = None

    @field_validator("device")
    @classmethod
    def uppercase_device(cls, v: str) -> str:
        """Normalise device string to uppercase (NPU, CPU, GPU)."""
        return v.upper()

    @model_validator(mode="after")
    def validate_custom_fns(self) -> "ModelEntry":
        """
        Ensure custom:<module.function> references are well-formed.
        Format: 'custom:some.module.function_name'
        """
        for field_name, val in [
            ("preprocess_fn", self.preprocess_fn),
            ("postprocess_fn", self.postprocess_fn),
        ]:
            if val.startswith("custom:"):
                ref = val.split(":", 1)[1]
                if "." not in ref:
                    raise ValueError(
                        f"[{self.name}] {field_name}='{val}' must be "
                        f"'custom:<module.function>' (e.g. custom:app.hooks.my_fn)"
                    )
        return self


class RegistryConfig(BaseModel):
    """
    Top-level config parsed from models.yaml.
    Ensures all model names are unique.
    """

    models: list[ModelEntry]

    @model_validator(mode="after")
    def unique_names(self) -> "RegistryConfig":
        """Reject duplicate model names at load time."""
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate model names in registry: {list(set(dupes))}")
        return self


def load_registry(config_path: str | Path) -> RegistryConfig:
    """Parse and validate models.yaml or models.json into a RegistryConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path.resolve()}")

    if path.suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raise ValueError(f"Unsupported registry format: '{path.suffix}'. Use .yaml or .json")

    cfg = RegistryConfig.model_validate(raw)
    log.info("Registry loaded: %d model(s).", len(cfg.models))
    for entry in cfg.models:
        log.info(
            "  %-24s task=%-10s device=%s  path=%s",
            entry.name, entry.task, entry.device, entry.path,
        )
    return cfg
