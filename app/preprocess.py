"""Preprocessing hooks: convert API request payloads into model inputs."""

from __future__ import annotations

from typing import Any


def build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """
    Construct a plain-text prompt from a list of chat messages.
    Used when the model does not expose a native chat template via openvino_genai.
    Most OV GenAI pipelines accept the raw messages list directly; this is the
    fallback for models loaded via the core OpenVINO IR path.
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


def build_prompt_from_input(input_: Any) -> str:
    """Normalise /v1/responses `input` field to a single string prompt."""
    if isinstance(input_, str):
        return input_
    if isinstance(input_, list):
        return build_prompt_from_messages([m if isinstance(m, dict) else m.model_dump() for m in input_])
    return str(input_)


def truncate_to_context(text: str, max_chars: int) -> str:
    """Hard-truncate input to avoid exceeding context window.
    Character-based heuristic; token-level truncation should be done by GenAI.
    """
    if len(text) > max_chars:
        return text[-max_chars:]  # keep the tail (most recent context)
    return text
