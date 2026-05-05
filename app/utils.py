"""Shared utilities: logging, SSE formatting, thread pool singleton, request IDs."""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from app import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pool – single shared instance for all blocking inference calls.
# ---------------------------------------------------------------------------
_thread_pool: Optional[ThreadPoolExecutor] = None


def get_thread_pool() -> ThreadPoolExecutor:
    """Initialize and return the singleton ThreadPoolExecutor."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=config.THREAD_POOL_SIZE,
            thread_name_prefix="ov-infer",
        )
        log.info("Thread pool initialized: max_workers=%d", config.THREAD_POOL_SIZE)
    return _thread_pool


def shutdown_thread_pool() -> None:
    """Shut down the singleton ThreadPoolExecutor and wait for running tasks."""
    global _thread_pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        log.info("Thread pool shut down.")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

SSE_DONE = "data: [DONE]\n\n"


def sse_chunk(payload: dict[str, Any]) -> str:
    """Serialize a dict as a single SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def make_stream_chunk(
    completion_id: str,
    model: str,
    content: str,
    finish_reason: Optional[str] = None,
) -> str:
    """Construct an OpenAI-compatible SSE chunk for chat completions."""
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if finish_reason is None else {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    return sse_chunk(payload)


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

def new_request_id() -> str:
    """Generate a short unique hex request identifier."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Structured request logging
# ---------------------------------------------------------------------------

def log_request(
    request_id: str,
    model: str,
    device: str,
    load_time_ms: float,
    inference_time_ms: float,
    total_time_ms: float,
    cache_hit: bool,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Emit one structured log line per inference request (INFO level)."""
    parts = (
        f"request_id={request_id} model={model} device={device} "
        f"load_ms={load_time_ms:.1f} infer_ms={inference_time_ms:.1f} "
        f"total_ms={total_time_ms:.1f} cache_hit={cache_hit} status={status}"
    )
    if error:
        parts += f" error={error!r}"
    log.info(parts)
