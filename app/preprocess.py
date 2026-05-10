"""Preprocessing: convert API payloads into raw model prompts."""

from __future__ import annotations

from typing import Any

_ROLE_TAGS = {"system": "system", "user": "user", "assistant": "assistant"}


def build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Build a ChatML-style prompt from chat message dicts.

    Uses the <|im_start|>/<|im_end|> format compatible with Qwen 2.5,
    Phi-3, and other ChatML-based models.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(parts)
    import logging
    logging.getLogger(__name__).info("🔍 PROMPT: len=%d chars", len(prompt))
    return prompt


def truncate_to_context(text: str, max_chars: int) -> str:
    """Hard-truncate prompt to stay within context window (keeps tail)."""
    return text[-max_chars:] if len(text) > max_chars else text


def normalize_stop_strings(stop: Any) -> list[str]:
    """Normalize OpenAI `stop` field (str | list[str] | None) to list[str]."""
    if isinstance(stop, list):
        return stop
    return [stop] if stop else []
