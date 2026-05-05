"""
FastAPI application entry-point.

Startup sequence:
  1. Validate NPU plugin availability (fail fast – no CPU fallback).
  2. Load and validate model registry from YAML/JSON.
  3. Register entries with ModelManager (lazy load on first request).
  4. Initialise thread pool.

Shutdown: drain thread pool gracefully.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import openvino as ov
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, utils
from app.model_manager import get_manager
from app.registry import load_registry
from app.schemas import ErrorDetail, ErrorResponse
import app.api as api_module

config.configure_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== OpenVINO NPU API | startup ===")
    _assert_npu_available()
    registry = load_registry(config.MODEL_CONFIG_PATH)
    get_manager().register_entries(registry.models)
    utils.get_thread_pool()
    log.info("Startup complete. %d model(s) registered.", len(registry.models))
    yield
    log.info("=== OpenVINO NPU API | shutdown ===")
    utils.shutdown_thread_pool()


def _assert_npu_available() -> None:
    """Fail fast if the NPU plugin is not loaded by OpenVINO Core."""
    core = ov.Core()
    available = core.available_devices
    log.info("OpenVINO available devices: %s", available)
    npu_found = any(
        d == config.NPU_DEVICE or d.startswith(config.NPU_DEVICE)
        for d in available
    )
    if not npu_found:
        raise RuntimeError(
            f"NPU device '{config.NPU_DEVICE}' not found in OpenVINO devices: {available}. "
            "Install the Intel NPU driver and openvino-npu plugin. "
            "This service does NOT fall back to CPU."
        )
    log.info("NPU plugin confirmed: '%s'", config.NPU_DEVICE)


app = FastAPI(
    title="OpenVINO NPU Inference API",
    description="OpenAI-compatible inference API for OpenVINO models on Intel NPU.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(api_module.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    body = ErrorResponse(error=ErrorDetail(message=str(exc), type="internal_server_error"))
    return JSONResponse(status_code=500, content=body.model_dump())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,      # single worker – NPU context is in-process
        log_config=None,
    )
