"""Preprocessing: convert API payloads into raw model prompts."""

from __future__ import annotations

from typing import Any

_ROLE_TAGS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Build a plain-text prompt from chat message dicts.

    GenAI LLMPipeline applies the model's chat template (jinja2) internally;
    this fallback is for models compiled via the raw Core path.
    """
    parts = []
    for msg in messages:
        tag = _ROLE_TAGS.get(msg.get("role", "user"), "")
        parts.append(f"{tag}\n{msg.get('content', '')}" if tag else msg.get("content", ""))
    parts.append("<|assistant|>")
    return "\n".join(parts)


def truncate_to_context(text: str, max_chars: int) -> str:
    """Hard-truncate prompt to stay within context window (keeps tail)."""
    return text[-max_chars:] if len(text) > max_chars else text


def normalize_stop_strings(stop: Any) -> list[str]:
    """Normalize OpenAI `stop` field (str | list[str] | None) to list[str]."""
    if isinstance(stop, list):
        return stop
    return [stop] if stop else []
