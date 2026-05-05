"""OpenAI adapter: translate between OpenAI and internal formats.

Isolation contract: NO imports from model_manager, pipeline, or registry.
"""

from __future__ import annotations

import time
import uuid

from app.schemas import (
    ChatCompletionChoice, ChatCompletionResponse,
    CompletionChoice, CompletionResponse,
    EmbeddingData, EmbeddingResponse,
    Message, OutputText, ResponseObject, ResponseOutput, ResponseRequest,
    UsageInfo,
)


def _token_estimate(text: str) -> int:
    """Coarse token count (~4 chars/token) for usage reporting."""
    return max(1, len(text) // 4)


def _usage(prompt: str, completion: str) -> UsageInfo:
    """Build UsageInfo from prompt and completion text."""
    p, c = _token_estimate(prompt), _token_estimate(completion)
    return UsageInfo(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


# Request adaptation

def messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert list[Message] to list[dict] for pipeline consumption."""
    return [m.model_dump() for m in messages]


def response_input_to_messages(body: ResponseRequest) -> list[dict]:
    """Normalize ResponseRequest.input into list[dict]."""
    if isinstance(body.input, str):
        return [{"role": "user", "content": body.input}]
    return [m.model_dump() if not isinstance(m, dict) else m for m in body.input]


def completion_input_to_prompt(prompt: str | list[str]) -> str:
    """Normalize completion prompt (str | list[str]) to a single string."""
    return "\n".join(prompt) if isinstance(prompt, list) else prompt


# Response construction

def make_chat_response(
    model: str, text: str, prompt: str, request_id: str | None = None,
) -> ChatCompletionResponse:
    """Build a non-streaming ChatCompletionResponse."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id or uuid.uuid4().hex[:16]}",
        model=model,
        choices=[ChatCompletionChoice(message=Message(role="assistant", content=text))],
        usage=_usage(prompt, text),
    )


def make_completion_response(
    model: str, text: str, prompt: str, request_id: str | None = None,
) -> CompletionResponse:
    """Build a non-streaming CompletionResponse."""
    return CompletionResponse(
        id=f"cmpl-{request_id or uuid.uuid4().hex[:16]}",
        model=model,
        choices=[CompletionChoice(text=text)],
        usage=_usage(prompt, text),
    )


def make_response_object(
    model: str, text: str, prompt: str, request_id: str | None = None,
) -> ResponseObject:
    """Build a ResponseObject for /v1/responses."""
    return ResponseObject(
        id=f"resp-{request_id or uuid.uuid4().hex[:16]}",
        created_at=int(time.time()), model=model,
        output=[ResponseOutput(content=[OutputText(text=text)])],
        usage=_usage(prompt, text),
    )


def make_embedding_response(
    model: str, vectors: list[list[float]], inputs: list[str],
) -> EmbeddingResponse:
    """Build an EmbeddingResponse from raw embedding vectors."""
    total = sum(_token_estimate(t) for t in inputs)
    return EmbeddingResponse(
        data=[EmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)],
        model=model,
        usage=UsageInfo(prompt_tokens=total, total_tokens=total),
    )
