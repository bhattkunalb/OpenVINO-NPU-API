"""
OpenAI adapter layer: translate between OpenAI request/response formats and
the internal pipeline format.

This module is intentionally isolated:
  - No FastAPI imports
  - No inference logic
  - Fully testable without a live model
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    OutputText,
    ResponseObject,
    ResponseOutput,
    ResponseRequest,
    UsageInfo,
)


# ---------------------------------------------------------------------------
# Request adaptation
# ---------------------------------------------------------------------------

def messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert list[Message] to list[dict] for pipeline consumption."""
    return [m.model_dump() for m in messages]


def response_input_to_messages(body: ResponseRequest) -> list[dict[str, Any]]:
    """
    Normalize the flexible ResponseRequest.input field into a list[dict].
    Handles str (single user turn) and list[Message] (full conversation).
    """
    if isinstance(body.input, str):
        return [{"role": "user", "content": body.input}]
    return [m.model_dump() if not isinstance(m, dict) else m for m in body.input]


# ---------------------------------------------------------------------------
# Response construction
# ---------------------------------------------------------------------------

def make_chat_response(
    model: str,
    text: str,
    prompt: str,
    request_id: Optional[str] = None,
) -> ChatCompletionResponse:
    """Build a non-streaming ChatCompletionResponse from raw generated text."""
    rid = request_id or uuid.uuid4().hex[:16]
    p_tok = _token_estimate(prompt)
    c_tok = _token_estimate(text)
    return ChatCompletionResponse(
        id=f"chatcmpl-{rid}",
        model=model,
        choices=[
            ChatCompletionChoice(
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
        ),
    )


def make_response_object(
    model: str,
    text: str,
    prompt: str,
    request_id: Optional[str] = None,
) -> ResponseObject:
    """Build a ResponseObject (for /v1/responses) from raw generated text."""
    rid = request_id or uuid.uuid4().hex[:16]
    p_tok = _token_estimate(prompt)
    c_tok = _token_estimate(text)
    return ResponseObject(
        id=f"resp-{rid}",
        created_at=int(time.time()),
        model=model,
        output=[ResponseOutput(content=[OutputText(text=text)])],
        usage=UsageInfo(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
        ),
    )


def make_embedding_response(
    model: str,
    vectors: list[list[float]],
    inputs: list[str],
) -> EmbeddingResponse:
    """Build an EmbeddingResponse from raw embedding vectors."""
    data = [EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(vectors)]
    total_tokens = sum(_token_estimate(t) for t in inputs)
    return EmbeddingResponse(
        data=data,
        model=model,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _token_estimate(text: str) -> int:
    """Coarse token count (≈4 chars/token) for usage reporting."""
    return max(1, len(text) // 4)
