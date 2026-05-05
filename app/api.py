"""
Route handlers, request validation, and SSE streaming.

Architecture:
  - Route handlers validate input and dispatch to pipeline.py via asyncio.to_thread().
  - adapter.py translates between OpenAI ↔ internal format (no inference logic here).
  - No business logic in routes: routes orchestrate only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app import postprocess, utils
from app.adapter import (
    make_chat_response,
    make_embedding_response,
    make_response_object,
    messages_to_dicts,
    response_input_to_messages,
)
from app.model_manager import get_manager
from app.pipeline import (
    run_embedding,
    run_generation,
    run_generation_stream,
)
from app.preprocess import build_prompt_from_messages, normalize_stop_strings
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    Message,
    ModelCard,
    ModelListResponse,
    ResponseObject,
    ResponseRequest,
)

log = logging.getLogger(__name__)
router = APIRouter()

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Bad Request"},
    404: {"model": ErrorResponse, "description": "Not Found"},
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
async def health() -> dict:
    """Return service health and model inventory."""
    manager = get_manager()
    return {
        "status": "ok",
        "loaded_models": manager.list_loaded(),
        "registered_models": manager.all_names(),
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@router.get("/v1/models", tags=["models"])
async def list_models() -> ModelListResponse:
    """Return an OpenAI-compatible list of all registered models."""
    cards = [ModelCard(id=name) for name in get_manager().all_names()]
    return ModelListResponse(data=cards)


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@router.post(
    "/v1/chat/completions",
    tags=["inference"],
    responses=_ERROR_RESPONSES,
)
async def chat_completions(
    body: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions with optional SSE streaming."""
    _require_gen_model(body.model)
    req_id = utils.new_request_id()

    if body.stream:
        return StreamingResponse(
            _stream_chat(req_id, body),
            media_type="text/event-stream",
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )

    return await _chat_non_stream(req_id, body)


async def _chat_non_stream(
    req_id: str, body: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Run non-streaming chat inference and return a complete response object."""
    manager = get_manager()
    t_total = time.perf_counter()
    stop = normalize_stop_strings(body.stop)
    msg_dicts = messages_to_dicts(body.messages)

    def _infer() -> tuple:
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        max_tok = body.max_tokens or cached.entry.max_tokens
        text, _, infer_ms = run_generation(
            cached, msg_dicts, max_tok, body.temperature, body.top_p, stop
        )
        return text, hit, load_ms, infer_ms, cached

    text, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_infer)
    total_ms = (time.perf_counter() - t_total) * 1000
    manager.record_inference(body.model, infer_ms)

    # Build a plain-text prompt approximation for token counting (adapter uses it)
    prompt = build_prompt_from_messages(msg_dicts)

    utils.log_request(req_id, body.model, cached.entry.device,
                      load_ms, infer_ms, total_ms, hit, "ok")
    return make_chat_response(body.model, text, prompt, req_id)


async def _stream_chat(
    req_id: str, body: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """
    SSE generator for streaming chat completions.

    OV GenAI streamer callback writes tokens into an asyncio.Queue;
    the event loop drains and yields them as SSE frames without blocking.
    """
    completion_id = f"chatcmpl-{req_id}"
    manager = get_manager()
    stop = normalize_stop_strings(body.stop)
    msg_dicts = messages_to_dicts(body.messages)
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _streamer_cb(token: str) -> bool:
        """Called by OV GenAI on each generated token. False = continue."""
        loop.call_soon_threadsafe(queue.put_nowait, token)
        return False

    load_start = time.perf_counter()

    def _infer_stream() -> tuple:
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - load_start) * 1000
        max_tok = body.max_tokens or cached.entry.max_tokens
        infer_ms = run_generation_stream(
            cached, msg_dicts, max_tok, body.temperature, body.top_p, stop, _streamer_cb
        )
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
        return hit, load_ms, infer_ms, cached

    future = loop.run_in_executor(utils.get_thread_pool(), _infer_stream)

    # Opening role delta (OpenAI protocol)
    yield utils.sse_chunk({
        "id": completion_id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": body.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    while True:
        token = await queue.get()
        if token is None:
            break
        cleaned = postprocess.clean_generation(token)
        if cleaned:
            yield utils.make_stream_chunk(completion_id, body.model, cleaned)

    yield utils.make_stream_chunk(completion_id, body.model, "", finish_reason="stop")
    yield utils.SSE_DONE

    hit, load_ms, infer_ms, cached = await future
    utils.log_request(req_id, body.model, cached.entry.device,
                      load_ms, infer_ms, 0.0, hit, "ok-stream")


# ---------------------------------------------------------------------------
# /v1/responses
# ---------------------------------------------------------------------------

@router.post(
    "/v1/responses",
    tags=["inference"],
    responses=_ERROR_RESPONSES,
)
async def responses(
    body: ResponseRequest,
) -> ResponseObject | StreamingResponse:
    """OpenAI Responses API style endpoint with optional SSE streaming."""
    _require_gen_model(body.model)
    req_id = utils.new_request_id()

    if body.stream:
        # Reuse chat streaming path with adapted messages
        adapted = ChatCompletionRequest(
            model=body.model,
            messages=[],   # not used; msg_dicts passed via adapter below
            max_tokens=body.max_output_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=True,
        )
        # Patch messages list with adapted input
        adapted_dicts = response_input_to_messages(body)
        adapted.messages = [Message(**d) for d in adapted_dicts]
        return StreamingResponse(
            _stream_chat(req_id, adapted),
            media_type="text/event-stream",
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )

    manager = get_manager()
    t_total = time.perf_counter()
    msg_dicts = response_input_to_messages(body)

    def _infer() -> tuple:
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        max_tok = body.max_output_tokens or cached.entry.max_tokens
        text, _, infer_ms = run_generation(
            cached, msg_dicts, max_tok, body.temperature, body.top_p, []
        )
        return text, hit, load_ms, infer_ms, cached

    text, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_infer)
    total_ms = (time.perf_counter() - t_total) * 1000
    manager.record_inference(body.model, infer_ms)

    prompt = build_prompt_from_messages(msg_dicts)

    utils.log_request(req_id, body.model, cached.entry.device,
                      load_ms, infer_ms, total_ms, hit, "ok")
    return make_response_object(body.model, text, prompt, req_id)


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

@router.post(
    "/v1/embeddings",
    tags=["inference"],
    responses=_ERROR_RESPONSES,
)
async def embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings. Only available when an embedding model is registered."""
    manager = get_manager()
    entry = manager.get_entry(body.model)
    if entry is None or entry.task != "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"'{body.model}' is not a registered embedding model. "
                   "Use GET /v1/models to list available models.",
        )

    req_id = utils.new_request_id()
    t_total = time.perf_counter()
    inputs = body.input if isinstance(body.input, list) else [body.input]

    def _infer() -> tuple:
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        vectors, infer_ms = run_embedding(cached, inputs)
        return vectors, hit, load_ms, infer_ms, cached

    vectors, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_infer)
    total_ms = (time.perf_counter() - t_total) * 1000
    manager.record_inference(body.model, infer_ms)
    utils.log_request(req_id, body.model, cached.entry.device,
                      load_ms, infer_ms, total_ms, hit, "ok")
    return make_embedding_response(body.model, vectors, inputs)


# ---------------------------------------------------------------------------
# Internal guards
# ---------------------------------------------------------------------------

def _require_gen_model(model_name: str) -> None:
    """Raise 404 if model not found, 400 if it is an embedding model."""
    entry = get_manager().get_entry(model_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Use GET /v1/models.",
        )
    if entry.task == "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"'{model_name}' is an embedding model. Use POST /v1/embeddings.",
        )
