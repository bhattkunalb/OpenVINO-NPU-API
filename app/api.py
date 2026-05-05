"""
Route definitions, request validation, and SSE streaming.

All inference dispatched to asyncio.to_thread() so blocking OpenVINO
calls never occupy the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import openvino_genai as ov_genai

from app import postprocess, preprocess, utils
from app.model_manager import get_manager
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    ErrorResponse,
    Message,
    ModelCard,
    ModelListResponse,
    OutputText,
    ResponseObject,
    ResponseOutput,
    ResponseRequest,
    UsageInfo,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
async def health() -> dict:
    manager = get_manager()
    return {
        "status": "ok",
        "loaded_models": manager.list_loaded(),
        "registered_models": manager.all_names(),
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@router.get("/v1/models", response_model=ModelListResponse, tags=["models"])
async def list_models() -> ModelListResponse:
    cards = [ModelCard(id=name) for name in get_manager().all_names()]
    return ModelListResponse(data=cards)


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions", tags=["inference"])
async def chat_completions(body: ChatCompletionRequest):
    _assert_gen_model(body.model)
    req_id = utils.new_request_id()
    if body.stream:
        return StreamingResponse(
            _stream_chat(req_id, body),
            media_type="text/event-stream",
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )
    return await _run_chat_inference(req_id, body)


async def _run_chat_inference(req_id: str, body: ChatCompletionRequest) -> ChatCompletionResponse:
    manager = get_manager()
    t_total = time.perf_counter()

    def _infer():
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        messages = [m.model_dump() for m in body.messages]
        prompt = preprocess.build_prompt_from_messages(messages)
        prompt = preprocess.truncate_to_context(prompt, cached.entry.context_window * 4)
        cfg = _build_gen_config(
            body.max_tokens or cached.entry.max_tokens,
            body.temperature, body.top_p,
            body.stop if isinstance(body.stop, list) else ([body.stop] if body.stop else []),
        )
        t_infer = time.perf_counter()
        with cached.lock:
            result = cached.pipeline.generate(prompt, cfg)
        infer_ms = (time.perf_counter() - t_infer) * 1000
        return result, hit, load_ms, infer_ms, cached

    result, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_infer)
    text = postprocess.clean_generation(_extract_text(result))
    raw_prompt = preprocess.build_prompt_from_messages([m.model_dump() for m in body.messages])
    p_tok = postprocess.estimate_token_count(raw_prompt)
    c_tok = postprocess.estimate_token_count(text)
    total_ms = (time.perf_counter() - t_total) * 1000

    manager.record_inference(body.model, infer_ms)
    utils.log_request(req_id, body.model, cached.entry.device_preference,
                      load_ms, infer_ms, total_ms, hit, "ok")

    return ChatCompletionResponse(
        id=f"chatcmpl-{req_id}",
        model=body.model,
        choices=[ChatCompletionChoice(message=Message(role="assistant", content=text))],
        usage=UsageInfo(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=p_tok + c_tok),
    )


async def _stream_chat(req_id: str, body: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """
    SSE generator. OV GenAI streamer callback bridges to asyncio.Queue so
    the event loop can yield chunks without blocking.
    """
    completion_id = f"chatcmpl-{req_id}"
    manager = get_manager()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _streamer_cb(token: str) -> bool:
        loop.call_soon_threadsafe(queue.put_nowait, token)
        return False  # True would stop generation early

    def _infer_stream():
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        messages = [m.model_dump() for m in body.messages]
        prompt = preprocess.build_prompt_from_messages(messages)
        prompt = preprocess.truncate_to_context(prompt, cached.entry.context_window * 4)
        cfg = _build_gen_config(
            body.max_tokens or cached.entry.max_tokens,
            body.temperature, body.top_p,
            body.stop if isinstance(body.stop, list) else ([body.stop] if body.stop else []),
        )
        t_infer = time.perf_counter()
        with cached.lock:
            cached.pipeline.generate(prompt, cfg, _streamer_cb)
        infer_ms = (time.perf_counter() - t_infer) * 1000
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
        return hit, load_ms, infer_ms, cached

    future = loop.run_in_executor(utils.get_thread_pool(), _infer_stream)

    # Role delta (OpenAI protocol)
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
    utils.log_request(req_id, body.model, cached.entry.device_preference,
                      load_ms, infer_ms, 0.0, hit, "ok-stream")


# ---------------------------------------------------------------------------
# /v1/responses
# ---------------------------------------------------------------------------

@router.post("/v1/responses", tags=["inference"])
async def responses(body: ResponseRequest):
    _assert_gen_model(body.model)
    req_id = utils.new_request_id()
    t_total = time.perf_counter()

    if body.stream:
        adapted = ChatCompletionRequest(
            model=body.model,
            messages=_normalise_response_input(body),
            max_tokens=body.max_output_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=True,
        )
        return StreamingResponse(
            _stream_chat(req_id, adapted),
            media_type="text/event-stream",
            headers={"X-Request-ID": req_id, "Cache-Control": "no-cache"},
        )

    manager = get_manager()

    def _infer():
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        prompt = preprocess.build_prompt_from_input(body.input)
        prompt = preprocess.truncate_to_context(prompt, cached.entry.context_window * 4)
        cfg = _build_gen_config(body.max_output_tokens or cached.entry.max_tokens,
                                body.temperature, body.top_p, [])
        t_infer = time.perf_counter()
        with cached.lock:
            result = cached.pipeline.generate(prompt, cfg)
        infer_ms = (time.perf_counter() - t_infer) * 1000
        return result, hit, load_ms, infer_ms, cached

    result, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_infer)
    text = postprocess.clean_generation(_extract_text(result))
    raw_prompt = preprocess.build_prompt_from_input(body.input)
    p_tok = postprocess.estimate_token_count(raw_prompt)
    c_tok = postprocess.estimate_token_count(text)
    total_ms = (time.perf_counter() - t_total) * 1000

    manager.record_inference(body.model, infer_ms)
    utils.log_request(req_id, body.model, cached.entry.device_preference,
                      load_ms, infer_ms, total_ms, hit, "ok")

    return ResponseObject(
        model=body.model,
        output=[ResponseOutput(content=[OutputText(text=text)])],
        usage=UsageInfo(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=p_tok + c_tok),
    )


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

@router.post("/v1/embeddings", tags=["inference"])
async def embeddings(body: EmbeddingRequest):
    manager = get_manager()
    entry = manager.get_entry(body.model)
    if entry is None or entry.task_type != "embedding":
        raise HTTPException(status_code=400,
                            detail=f"'{body.model}' is not a registered embedding model.")

    req_id = utils.new_request_id()
    t_total = time.perf_counter()
    inputs = body.input if isinstance(body.input, list) else [body.input]

    def _infer():
        t0 = time.perf_counter()
        cached, hit = manager.get_cached(body.model)
        load_ms = (time.perf_counter() - t0) * 1000
        t_infer = time.perf_counter()
        with cached.lock:
            if hasattr(cached.pipeline, "infer"):
                vectors = cached.pipeline.infer(inputs)
            elif hasattr(cached.pipeline, "embed"):
                vectors = cached.pipeline.embed(inputs)
            else:
                # Raw OV compiled model fallback
                results = []
                for text in inputs:
                    req = cached.pipeline.create_infer_request()
                    inp = cached.pipeline.input(0)
                    import numpy as np
                    req.infer({inp.any_name: np.array([[ord(c) for c in text[:512]]])})
                    results.append(req.get_output_tensor(0).data.flatten().tolist())
                vectors = results
        infer_ms = (time.perf_counter() - t_infer) * 1000
        return vectors, hit, load_ms, infer_ms, cached

    vectors, hit, load_ms, infer_ms, cached = await asyncio.to_thread(_infer)
    total_ms = (time.perf_counter() - t_total) * 1000
    manager.record_inference(body.model, infer_ms)
    utils.log_request(req_id, body.model, cached.entry.device_preference,
                      load_ms, infer_ms, total_ms, hit, "ok")

    data = [EmbeddingData(index=i, embedding=list(v)) for i, v in enumerate(vectors)]
    total_tokens = sum(postprocess.estimate_token_count(t) for t in inputs)
    return EmbeddingResponse(
        data=data, model=body.model,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_gen_model(model_name: str) -> None:
    entry = get_manager().get_entry(model_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    if entry.task_type == "embedding":
        raise HTTPException(status_code=400,
                            detail=f"'{model_name}' is an embedding model; use /v1/embeddings.")


def _build_gen_config(
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    stop_strings: list[str],
) -> ov_genai.GenerationConfig:
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    if temperature is not None:
        cfg.temperature = temperature
    if top_p is not None:
        cfg.top_p = top_p
    if stop_strings:
        try:
            cfg.stop_strings = set(stop_strings)
        except AttributeError:
            pass  # older SDK – skip silently
    return cfg


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "texts"):
        return result.texts[0] if result.texts else ""
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _normalise_response_input(body: ResponseRequest) -> list[Message]:
    if isinstance(body.input, str):
        return [Message(role="user", content=body.input)]
    return list(body.input)
