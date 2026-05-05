"""
FastAPI application entry-point.

Startup sequence:
  1. Configure logging.
  2. Validate NPU plugin availability (fail fast – no CPU fallback).
  3. Parse and validate models.yaml registry.
  4. Register model entries with ModelManager (lazy compilation on first request).
  5. Initialize thread pool.

Shutdown: drain thread pool, release resources.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import openvino as ov
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import config, utils
from app.model_manager import get_manager
from app.registry import load_registry
from app.schemas import ErrorDetail, ErrorResponse
import app.api as api_module

config.configure_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of the inference service."""
    log.info("=== OpenVINO NPU API | startup ===")
    _assert_npu_available()
    registry = load_registry(config.MODEL_CONFIG_PATH)
    get_manager().register_entries(registry.models)
    utils.get_thread_pool()
    log.info("Startup complete. %d model(s) registered (lazy load).", len(registry.models))
    yield
    log.info("=== OpenVINO NPU API | shutdown ===")
    utils.shutdown_thread_pool()


def _assert_npu_available() -> None:
    """Abort startup if the NPU plugin is not present in OpenVINO Core."""
    core = ov.Core()
    available = core.available_devices
    log.info("OpenVINO available devices: %s", available)
    npu_found = any(
        d == config.NPU_DEVICE or d.startswith(config.NPU_DEVICE)
        for d in available
    )
    if not npu_found:
        raise RuntimeError(
            f"NPU device '{config.NPU_DEVICE}' not found. "
            f"Available devices: {available}. "
            "Install the Intel NPU driver and openvino-intel-npu plugin. "
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


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce OPENVINO_API_KEY if configured in environment."""

    async def dispatch(self, request: Request, call_next):
        if not config.API_KEY:
            return await call_next(request)

        # Allow health and docs without key
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        expected = f"Bearer {config.API_KEY}"
        if not auth_header or auth_header != expected:
            error_body = {
                "error": {
                    "message": "Invalid or missing API key.",
                    "type": "invalid_request_error"
                }
            }
            return JSONResponse(status_code=401, content=error_body)
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)
app.include_router(api_module.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: return OpenAI-compatible error JSON for any unhandled exception."""
    log.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True,
    )
    body = ErrorResponse(error=ErrorDetail(message=str(exc), type="internal_server_error"))
    return JSONResponse(status_code=500, content=body.model_dump())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=1,       # NPU context is in-process; single worker is required
        log_config=None,
    )
