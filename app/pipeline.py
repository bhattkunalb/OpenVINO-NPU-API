"""Pipeline execution: preprocess → infer → postprocess.

All inference passes through run_generation() and run_embedding().
Blocking calls; callers MUST dispatch via asyncio.to_thread().
"""

from __future__ import annotations

import time
from typing import Any

import openvino_genai as ov_genai

from app import postprocess, preprocess
from app.model_manager import CachedModel


def build_gen_config(
    max_tokens: int, temperature: float | None,
    top_p: float | None, stop_strings: list[str],
) -> ov_genai.GenerationConfig:
    """Construct an OV GenAI GenerationConfig from OpenAI-style params."""
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    do_sample = False
    if temperature is not None and temperature > 0.0:
        cfg.temperature = temperature
        do_sample = True
    if top_p is not None and top_p < 1.0:
        cfg.top_p = top_p
    cfg.do_sample = do_sample
    
    if stop_strings:
        try:
            cfg.stop_strings = set(stop_strings)
        except AttributeError:
            pass  # older SDK lacks stop_strings
    return cfg


def run_generation(
    cached: CachedModel, prompt_or_messages: str | list[dict[str, Any]],
    max_tokens: int, temperature: float | None,
    top_p: float | None, stop_strings: list[str],
) -> tuple[str, float, float]:
    """Blocking text generation. Returns (text, 0.0, infer_ms)."""
    prompt = _resolve_prompt(prompt_or_messages, cached)
    cfg = build_gen_config(max_tokens, temperature, top_p, stop_strings)
    t0 = time.perf_counter()
    with cached.lock:
        result = cached.pipeline.generate(prompt, cfg)
    infer_ms = (time.perf_counter() - t0) * 1000
    text = postprocess.clean_generation(_to_str(result))
    if stop_strings:
        text = postprocess.enforce_stop_strings(text, stop_strings)
    return text, 0.0, infer_ms


def run_generation_stream(
    cached: CachedModel, prompt_or_messages: str | list[dict[str, Any]],
    max_tokens: int, temperature: float | None,
    top_p: float | None, stop_strings: list[str], streamer_cb: Any,
) -> float:
    """Blocking streaming generation. Returns infer_ms."""
    prompt = _resolve_prompt(prompt_or_messages, cached)
    cfg = build_gen_config(max_tokens, temperature, top_p, stop_strings)
    t0 = time.perf_counter()
    with cached.lock:
        cached.pipeline.generate(prompt, cfg, streamer_cb)
    return (time.perf_counter() - t0) * 1000


def run_embedding(cached: CachedModel, inputs: list[str]) -> tuple[list[list[float]], float]:
    """Blocking embedding inference. Returns (vectors, infer_ms)."""
    t0 = time.perf_counter()
    with cached.lock:
        fn = getattr(cached.pipeline, "infer", None) or getattr(cached.pipeline, "embed", None)
        if fn:
            vectors: list[list[float]] = fn(inputs)
        else:
            raise RuntimeError(f"Embedding model '{cached.entry.name}' has no infer/embed method.")
    return vectors, (time.perf_counter() - t0) * 1000


def _resolve_prompt(
    data: str | list[dict[str, Any]], cached: CachedModel
) -> str | list[dict[str, Any]]:
    """Standardize input to a context-clipped prompt string or raw message list.
    
    If data is a list of messages, pass it through directly to let LLMPipeline
    apply the model's native chat template (jinja2).
    """
    if isinstance(data, list):
        return data
    return preprocess.truncate_to_context(data, cached.entry.context_length * 4)


def _to_str(result: Any) -> str:
    """Unify OV GenAI result shapes into a plain string."""
    if isinstance(result, str):
        return result
    if hasattr(result, "texts"):
        return result.texts[0] if result.texts else ""
    return result.text if hasattr(result, "text") else str(result)
