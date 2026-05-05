"""
Pipeline execution contract: preprocess → infer → postprocess.

All model inference passes through run_generation() and run_embedding().
No business logic here – only the three-stage pipeline contract.
Blocking calls; callers MUST dispatch via asyncio.to_thread() or ThreadPoolExecutor.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import openvino_genai as ov_genai

from app import postprocess, preprocess
from app.model_manager import CachedModel


def build_gen_config(
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    stop_strings: list[str],
) -> ov_genai.GenerationConfig:
    """Construct an OV GenAI GenerationConfig from OpenAI-style parameters."""
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
            pass  # older SDK – stop_strings not supported, skip silently
    return cfg


def run_generation(
    cached: CachedModel,
    prompt_or_messages: str | list[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    stop_strings: list[str],
) -> tuple[str, float, float]:
    """
    Blocking text generation: preprocess → infer → postprocess.

    Returns:
        (output_text, load_time_ms, infer_time_ms)
    """
    prompt = _get_prompt(prompt_or_messages, cached)
    cfg = build_gen_config(max_tokens, temperature, top_p, stop_strings)

    t_infer = time.perf_counter()
    with cached.lock:
        result = cached.pipeline.generate(prompt, cfg)
    infer_ms = (time.perf_counter() - t_infer) * 1000

    text = postprocess.clean_generation(_extract_text(result))
    return text, 0.0, infer_ms


def run_generation_stream(
    cached: CachedModel,
    prompt_or_messages: str | list[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    stop_strings: list[str],
    streamer_cb: Any,
) -> float:
    """
    Blocking streaming generation: preprocess → infer(cb) → done.
    """
    prompt = _get_prompt(prompt_or_messages, cached)
    cfg = build_gen_config(max_tokens, temperature, top_p, stop_strings)

    t_infer = time.perf_counter()
    with cached.lock:
        cached.pipeline.generate(prompt, cfg, streamer_cb)
    return (time.perf_counter() - t_infer) * 1000


def _get_prompt(data: str | list[dict[str, Any]], cached: CachedModel) -> str:
    """Standardize input to a context-clipped prompt string."""
    if isinstance(data, str):
        prompt = data
    else:
        prompt = preprocess.build_prompt_from_messages(data)

    return preprocess.truncate_to_context(
        prompt, cached.entry.context_length * 4
    )


def run_embedding(
    cached: CachedModel,
    inputs: list[str],
) -> tuple[list[list[float]], float]:
    """
    Blocking embedding inference.

    Returns:
        (list of embedding vectors, infer_time_ms)
    """
    t_infer = time.perf_counter()
    with cached.lock:
        if hasattr(cached.pipeline, "infer"):
            vectors: list[list[float]] = cached.pipeline.infer(inputs)
        elif hasattr(cached.pipeline, "embed"):
            vectors = cached.pipeline.embed(inputs)
        else:
            # Raw Core compiled model fallback for models without GenAI EmbeddingModel
            vectors = _infer_core_embedding(cached.pipeline, inputs)
    infer_ms = (time.perf_counter() - t_infer) * 1000
    return vectors, infer_ms


def _infer_core_embedding(model: Any, inputs: list[str]) -> list[list[float]]:
    """Run embedding via raw OV Core compiled model when GenAI API unavailable."""
    results: list[list[float]] = []
    for text in inputs:
        # Minimal char-code tokenization – replace with openvino-tokenizers in production
        char_ids = [ord(c) for c in text[:512]]
        req = model.create_infer_request()
        inp = model.input(0)
        req.infer({inp.any_name: np.array([char_ids], dtype=np.int32)})
        results.append(req.get_output_tensor(0).data.flatten().tolist())
    return results


def _extract_text(result: Any) -> str:
    """Unify different OV GenAI result shapes into a plain string."""
    if isinstance(result, str):
        return result
    if hasattr(result, "texts"):
        return result.texts[0] if result.texts else ""
    if hasattr(result, "text"):
        return result.text
    return str(result)
