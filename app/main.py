"""FastAPI application entry-point.

Startup: validate NPU → load registry → initialize thread pool.
Shutdown: drain thread pool.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import openvino as ov
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import api, config, utils
from app.model_manager import get_manager
from app.registry import load_registry
from app.schemas import ErrorDetail, ErrorResponse

config.configure_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of the inference service."""
    log.info("=== OpenVINO NPU API | startup ===")
    _assert_npu()
    get_manager().register_entries(load_registry(config.MODEL_CONFIG_PATH).models)
    utils.get_thread_pool()
    log.info("Startup complete.")
    yield
    log.info("=== shutdown ===")
    utils.shutdown_thread_pool()


def _assert_npu() -> None:
    """Abort startup if NPU plugin is absent."""
    available = ov.Core().available_devices
    log.info("OpenVINO devices: %s", available)
    if not any(d == config.NPU_DEVICE or d.startswith(config.NPU_DEVICE) for d in available):
        raise RuntimeError(
            f"NPU '{config.NPU_DEVICE}' not found. Available: {available}. "
            "Install intel-npu-driver + openvino-intel-npu. No CPU fallback."
        )
    log.info("NPU confirmed: '%s'", config.NPU_DEVICE)


app = FastAPI(
    title="OpenVINO NPU Inference API",
    description="OpenAI-compatible inference on Intel NPU.",
    version="1.0.0", docs_url="/docs", redoc_url="/redoc", lifespan=lifespan,
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Enforce OPENVINO_API_KEY if set; skip for /health and /docs."""

    async def dispatch(self, request: Request, call_next):
        if not config.API_KEY or request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {config.API_KEY}":
            return JSONResponse(status_code=401, content={"error": {"message": "Invalid API key.", "type": "auth_error"}})
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)
app.include_router(api.router)


@app.exception_handler(Exception)
async def global_exc(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: return OpenAI-compatible error JSON."""
    log.error("Unhandled %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    body = ErrorResponse(error=ErrorDetail(message=str(exc), type="internal_server_error"))
    return JSONResponse(status_code=500, content=body.model_dump())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, workers=1, log_config=None)
