"""Environment variable parsing, path resolution, and logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Path to model registry YAML
MODEL_CONFIG_PATH: Path = Path(_env("CONFIG_PATH", "models.yaml"))

# Default NPU device string (override with NPU_DEVICE_STRING env var)
NPU_DEVICE: str = _env("NPU_DEVICE_STRING", "NPU")

# Log level
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO").upper()

# Thread pool for blocking inference
THREAD_POOL_SIZE: int = int(_env("OPENVINO_API_THREAD_POOL_SIZE", "4"))

# Uvicorn bind
HOST: str = _env("OPENVINO_API_HOST", "0.0.0.0")
PORT: int = int(_env("OPENVINO_API_PORT", "4647"))

# Optional API Key (open by default if empty)
API_KEY: str | None = os.environ.get("OPENVINO_API_KEY")


def configure_logging() -> None:
    """Apply global structured log format."""
    numeric = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s │ %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
