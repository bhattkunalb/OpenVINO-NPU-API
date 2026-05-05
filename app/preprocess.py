"""Preprocessing hooks: convert API request payloads into raw model prompts."""

from __future__ import annotations

from typing import Any, Union

from app.schemas import Message


def build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """
    Construct a plain-text prompt from a list of chat message dicts.

    GenAI LLMPipeline accepts this string directly; it applies the model's
    built-in chat template (minja) internally during generation.
    This fallback is used for models compiled via the raw Core path.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        else:
            parts.append(content)
    parts.append("<|assistant|>")
    return "\n".join(parts)


def build_prompt_from_input(input_: Union[str, list[Message]]) -> str:
    """Normalize /v1/responses `input` field to a single string prompt."""
    if isinstance(input_, str):
        return input_
    # list[Message] – convert each to dict then build chat prompt
    return build_prompt_from_messages(
        [m if isinstance(m, dict) else m.model_dump() for m in input_]
    )


def truncate_to_context(text: str, max_chars: int) -> str:
    """
    Hard-truncate prompt to avoid exceeding the context window.

    Character-based heuristic (≈4 chars/token); GenAI performs token-level
    clipping internally – this is a safety net for very long inputs.
    Keeps the tail so that the most-recent context is preserved.
    """
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def normalize_stop_strings(stop: Any) -> list[str]:
    """Normalize the OpenAI `stop` field (str | list[str] | None) to list[str]."""
    if isinstance(stop, list):
        return stop
    if stop:
        return [stop]
    return []
