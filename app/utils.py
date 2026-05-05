"""Shared utilities: logging helpers, SSE formatting, thread pool singleton."""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator, Optional

from app import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pool – single shared instance for all blocking inference calls.
# OpenVINO inference is not thread-safe per-pipeline; ModelManager enforces
# per-model locking, so max_workers here controls overall parallelism budget.
# ---------------------------------------------------------------------------
_thread_pool: Optional[ThreadPoolExecutor] = None


def get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=config.THREAD_POOL_SIZE,
            thread_name_prefix="ov-infer",
        )
        log.info("Thread pool initialised: max_workers=%d", config.THREAD_POOL_SIZE)
    return _thread_pool


def shutdown_thread_pool() -> None:
    global _thread_pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        log.info("Thread pool shut down.")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse_chunk(payload: dict[str, Any]) -> str:
    """Format a single SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


SSE_DONE = "data: [DONE]\n\n"


def make_stream_chunk(
    completion_id: str,
    model: str,
    content: str,
    finish_reason: Optional[str] = None,
) -> str:
    payload = {
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
    return uuid.uuid4().hex


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
) -> None:
    log.info(
        "request_id=%s model=%s device=%s load_ms=%.1f infer_ms=%.1f total_ms=%.1f "
        "cache_hit=%s status=%s",
        request_id,
        model,
        device,
        load_time_ms,
        inference_time_ms,
        total_time_ms,
        cache_hit,
        status,
    )
