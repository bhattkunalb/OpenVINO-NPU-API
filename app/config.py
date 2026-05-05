"""Environment variable parsing and config path resolution."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Config file path – override with OPENVINO_API_MODEL_CONFIG
MODEL_CONFIG_PATH: Path = Path(_env("OPENVINO_API_MODEL_CONFIG", "models.yaml"))

# NPU device string – override if your platform uses a different plugin name
NPU_DEVICE: str = _env("OPENVINO_API_DEVICE", "NPU")

# Log level
LOG_LEVEL: str = _env("OPENVINO_API_LOG_LEVEL", "INFO").upper()

# Thread pool size for blocking inference calls
THREAD_POOL_SIZE: int = int(_env("OPENVINO_API_THREAD_POOL_SIZE", "4"))

# Host/port (used by Uvicorn entrypoint in main.py)
HOST: str = _env("OPENVINO_API_HOST", "0.0.0.0")
PORT: int = int(_env("OPENVINO_API_PORT", "8000"))


def configure_logging() -> None:
    """Apply structured log format globally."""
    numeric = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s │ %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
