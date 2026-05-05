"""Registry: load, validate, and resolve model entries from YAML/JSON config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

log = logging.getLogger(__name__)

TaskType = Literal["chat", "response", "embedding"]
PreprocessorType = Literal["default_genai", "custom"]
PostprocessorType = Literal["default_genai", "custom"]


class ModelEntry(BaseModel):
    name: str
    path: str  # absolute path OR HF-style "org/repo"
    task_type: TaskType
    device_preference: str = "NPU"
    max_tokens: int = 2048
    context_window: int = 4096
    preprocessor: PreprocessorType = "default_genai"
    postprocessor: PostprocessorType = "default_genai"
    preprocessor_path: Optional[str] = None   # required when preprocessor=="custom"
    postprocessor_path: Optional[str] = None  # required when postprocessor=="custom"

    @field_validator("device_preference")
    @classmethod
    def uppercase_device(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def check_custom_paths(self) -> "ModelEntry":
        if self.preprocessor == "custom" and not self.preprocessor_path:
            raise ValueError(f"[{self.name}] preprocessor_path required when preprocessor='custom'")
        if self.postprocessor == "custom" and not self.postprocessor_path:
            raise ValueError(f"[{self.name}] postprocessor_path required when postprocessor='custom'")
        return self


class RegistryConfig(BaseModel):
    models: list[ModelEntry]

    @model_validator(mode="after")
    def unique_names(self) -> "RegistryConfig":
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate model names detected in registry.")
        return self


def load_registry(config_path: str | Path) -> RegistryConfig:
    """Parse models.yaml or models.json and return validated RegistryConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")

    raw: dict
    if path.suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")

    cfg = RegistryConfig.model_validate(raw)
    log.info("Registry loaded: %d model(s) defined.", len(cfg.models))
    for entry in cfg.models:
        log.info("  [%s] task=%s device=%s path=%s", entry.name, entry.task_type, entry.device_preference, entry.path)
    return cfg
