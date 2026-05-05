"""Shared utilities: SSE formatting, thread pool, request IDs, structured logging."""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app import config

log = logging.getLogger(__name__)

_thread_pool: ThreadPoolExecutor | None = None


def get_thread_pool() -> ThreadPoolExecutor:
    """Return the singleton ThreadPoolExecutor for blocking inference."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=config.THREAD_POOL_SIZE, thread_name_prefix="ov-infer"
        )
        log.info("Thread pool: max_workers=%d", config.THREAD_POOL_SIZE)
    return _thread_pool


def shutdown_thread_pool() -> None:
    """Drain and release the thread pool."""
    global _thread_pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None


# SSE helpers

SSE_DONE = "data: [DONE]\n\n"


def sse_event(payload: dict[str, Any]) -> str:
    """Serialize a dict as a single SSE data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def make_stream_chunk(
    completion_id: str, model: str, content: str, finish_reason: str | None = None
) -> str:
    """Build an OpenAI-compatible SSE chunk for chat completion streaming."""
    return sse_event({
        "id": completion_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if finish_reason is None else {},
            "finish_reason": finish_reason, "logprobs": None,
        }],
    })


def new_request_id() -> str:
    """Generate a short unique hex request identifier."""
    return uuid.uuid4().hex[:16]


def log_request(
    request_id: str, model: str, device: str,
    load_ms: float, infer_ms: float, total_ms: float,
    cache_hit: bool, status: str, error: str | None = None,
) -> None:
    """Emit one structured log line per inference request."""
    msg = (
        f"request_id={request_id} model={model} device={device} "
        f"load_ms={load_ms:.1f} infer_ms={infer_ms:.1f} total_ms={total_ms:.1f} "
        f"cache_hit={cache_hit} status={status}"
    )
    log.info("%s error=%r", msg, error) if error else log.info(msg)
