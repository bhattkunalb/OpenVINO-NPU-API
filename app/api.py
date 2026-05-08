"""Route handlers, request validation, and SSE streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app import postprocess, utils
from app.adapter import (
    completion_input_to_prompt, make_chat_response, make_completion_response,
    make_embedding_response, make_response_object, messages_to_dicts,
    response_input_to_messages,
)
from app.model_manager import get_manager
from app.pipeline import run_embedding, run_generation, run_generation_stream
from app.preprocess import build_prompt_from_messages, normalize_stop_strings
from app.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest,
    CompletionResponse, EmbeddingRequest, EmbeddingResponse, ErrorResponse,
    Message, ModelCard, ModelListResponse, ResponseObject, ResponseRequest,
)

log = logging.getLogger(__name__)
router = APIRouter()

STREAM_TYPE = "text/event-stream"

_ERR = {
    400: {"model": ErrorResponse, "description": "Bad Request"},
    404: {"model": ErrorResponse, "description": "Not Found"},
}


# Health & models

@router.get("/health", tags=["system"])
async def health() -> dict:
    """Service health and model inventory."""
    mgr = get_manager()
    return {
        "status": "ok",
        "loaded_models": mgr.list_loaded(),
        "registered_models": mgr.all_names(),
    }


@router.get("/v1/models", tags=["models"])
async def list_models() -> ModelListResponse:
    """OpenAI-compatible model list."""
    return ModelListResponse(data=[ModelCard(id=n) for n in get_manager().all_names()])


# Shared inference helper

async def _infer(model: str, msg_dicts: list[dict], body: object) -> tuple:
    """Run non-streaming generation for any endpoint.
    Returns (text, hit, load_ms, infer_ms, total_ms, cached).
    """
    mgr = get_manager()
    t0 = time.perf_counter()
    stop = normalize_stop_strings(getattr(body, "stop", None))
    max_tok_attr = getattr(body, "max_tokens", None) or getattr(body, "max_output_tokens", None)

    def _run() -> tuple:
        t_load = time.perf_counter()
        cached, hit = mgr.get_cached(model)
        load_ms = (time.perf_counter() - t_load) * 1000
        max_tok = max_tok_attr or cached.entry.max_tokens
        text, _, infer_ms = run_generation(
            cached, msg_dicts, max_tok,
            getattr(body, "temperature", 1.0), getattr(body, "top_p", 1.0), stop,
        )
        return text, hit, load_ms, infer_ms, cached

    text, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_run)
    total_ms = (time.perf_counter() - t0) * 1000
    mgr.record_inference(model, infer_ms)
    return text, hit, load_ms, infer_ms, total_ms, cached


# /v1/chat/completions

@router.post("/v1/chat/completions", tags=["inference"], responses=_ERR, response_model=None)
async def chat_completions(
    body: ChatCompletionRequest
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions with optional SSE streaming."""
    _require_gen(body.model)
    req_id = utils.new_request_id()
    if body.stream:
        return StreamingResponse(
            _stream_chat(req_id, body), media_type=STREAM_TYPE,
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )
    msg_dicts = messages_to_dicts(body.messages)
    text, hit, load_ms, infer_ms, total_ms, cached = await _infer(
        body.model, msg_dicts, body
    )
    prompt = build_prompt_from_messages(msg_dicts)
    utils.log_request(
        req_id, body.model, cached.entry.device,
        load_ms, infer_ms, total_ms, hit, "ok"
    )
    return make_chat_response(body.model, text, prompt, req_id)


async def _stream_chat(
    req_id: str,
    body: ChatCompletionRequest,
    override_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """SSE generator for streaming chat completions."""
    cid = f"chatcmpl-{req_id}"
    mgr = get_manager()
    stop = normalize_stop_strings(body.stop)
    msg_dicts = messages_to_dicts(body.messages)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _cb(token: str) -> bool:
        loop.call_soon_threadsafe(queue.put_nowait, token)
        return False

    t0 = time.perf_counter()
    def _run() -> tuple:
        cached, hit = mgr.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        max_tok = body.max_tokens or cached.entry.max_tokens
        infer_ms = run_generation_stream(
            cached, override_prompt or msg_dicts, max_tok,
            body.temperature, body.top_p, stop, _cb,
        )
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return hit, load_ms, infer_ms, cached

    future = loop.run_in_executor(utils.get_thread_pool(), _run)

    yield utils.sse_event({
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": body.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    stop_manager = postprocess.StreamStopManager(stop)
    while True:
        token = await queue.get()
        if token is None:
            break

        safe_token = stop_manager.process_token(token)
        if safe_token:
            cleaned = postprocess.clean_generation(safe_token)
            if cleaned:
                yield utils.make_stream_chunk(cid, body.model, cleaned)

        if stop_manager.stopped:
            break

    final_text = stop_manager.flush()
    if final_text:
        cleaned = postprocess.clean_generation(final_text)
        if cleaned:
            yield utils.make_stream_chunk(cid, body.model, cleaned)

    yield utils.make_stream_chunk(cid, body.model, "", finish_reason="stop")
    yield utils.SSE_DONE

    hit, load_ms, infer_ms, cached = await future
    utils.log_request(
        req_id, body.model, cached.entry.device,
        load_ms, infer_ms, 0.0, hit, "ok-stream"
    )


# /v1/completions

@router.post("/v1/completions", tags=["inference"], responses=_ERR, response_model=None)
async def completions(body: CompletionRequest) -> CompletionResponse | StreamingResponse:
    """Standard text completions with optional SSE streaming."""
    _require_gen(body.model)
    req_id = utils.new_request_id()
    prompt = completion_input_to_prompt(body.prompt)
    if body.stream:
        adapted = ChatCompletionRequest(
            model=body.model, messages=[], max_tokens=body.max_tokens,
            temperature=body.temperature, top_p=body.top_p,
            stream=True, stop=body.stop,
        )
        return StreamingResponse(
            _stream_chat(req_id, adapted, override_prompt=prompt),
            media_type=STREAM_TYPE,
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )
    text, hit, load_ms, infer_ms, total_ms, cached = await _infer(
        body.model, prompt, body
    )
    utils.log_request(
        req_id, body.model, cached.entry.device,
        load_ms, infer_ms, total_ms, hit, "ok"
    )
    return make_completion_response(body.model, text, prompt, req_id)


# /v1/responses

@router.post("/v1/responses", tags=["inference"], responses=_ERR, response_model=None)
async def responses(body: ResponseRequest) -> ResponseObject | StreamingResponse:
    """OpenAI Responses API endpoint with optional SSE streaming."""
    _require_gen(body.model)
    req_id = utils.new_request_id()
    if body.stream:
        adapted_dicts = response_input_to_messages(body)
        adapted = ChatCompletionRequest(
            model=body.model,
            messages=[Message(**d) for d in adapted_dicts],
            max_tokens=body.max_output_tokens,
            temperature=body.temperature,
            top_p=body.top_p, stream=True,
        )
        return StreamingResponse(
            _stream_chat(req_id, adapted), media_type=STREAM_TYPE,
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )
    msg_dicts = response_input_to_messages(body)
    text, hit, load_ms, infer_ms, total_ms, cached = await _infer(
        body.model, msg_dicts, body
    )
    prompt = build_prompt_from_messages(msg_dicts)
    utils.log_request(
        req_id, body.model, cached.entry.device,
        load_ms, infer_ms, total_ms, hit, "ok"
    )
    return make_response_object(body.model, text, prompt, req_id)


# /v1/embeddings

@router.post("/v1/embeddings", tags=["inference"], responses=_ERR)
async def embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings (only for registered embedding models)."""
    mgr = get_manager()
    entry = mgr.get_entry(body.model)
    if entry is None or entry.task != "embedding":
        raise HTTPException(
            400, f"'{body.model}' is not an embedding model. Use GET /v1/models."
        )
    req_id = utils.new_request_id()
    t0 = time.perf_counter()
    inputs = body.input if isinstance(body.input, list) else [body.input]

    def _run() -> tuple:
        cached, hit = mgr.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        vectors, infer_ms = run_embedding(cached, inputs)
        return vectors, hit, load_ms, infer_ms, cached

    vectors, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_run)
    total_ms = (time.perf_counter() - t0) * 1000
    mgr.record_inference(body.model, infer_ms)
    utils.log_request(
        req_id, body.model, cached.entry.device,
        load_ms, infer_ms, total_ms, hit, "ok"
    )
    return make_embedding_response(body.model, vectors, inputs)


def _require_gen(model: str) -> None:
    """Raise 404/400 for missing or embedding-only models."""
    entry = get_manager().get_entry(model)
    if entry is None:
        raise HTTPException(404, f"Model '{model}' not found. Use GET /v1/models.")
    if entry.task == "embedding":
        raise HTTPException(
            400, f"'{model}' is an embedding model. Use POST /v1/embeddings."
        )
