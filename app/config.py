"""Environment-variable configuration and logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


MODEL_CONFIG_PATH: Path = Path(os.environ.get("CONFIG_PATH", "models.yaml"))
NPU_DEVICE: str = os.environ.get("NPU_DEVICE_STRING", "NPU")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
THREAD_POOL_SIZE: int = int(os.environ.get("OPENVINO_API_THREAD_POOL_SIZE", "4"))
HOST: str = os.environ.get("OPENVINO_API_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("OPENVINO_API_PORT", "4647"))
API_KEY: Optional[str] = os.environ.get("OPENVINO_API_KEY")


def configure_logging() -> None:
    """Apply global structured log format."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
