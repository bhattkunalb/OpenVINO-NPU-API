"""OpenAI-compatible Pydantic request/response schemas."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class UsageInfo(BaseModel):
    """Token usage counters."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ErrorDetail(BaseModel):
    """Inner error object."""
    message: str
    type: str = "server_error"
    param: str | None = None
    code: str | None = None

class ErrorResponse(BaseModel):
    """Top-level error envelope."""
    error: ErrorDetail

class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request."""
    model: str
    messages: list[Message]
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(1, ge=1, le=1)

class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""
    index: int = 0
    message: Message
    finish_reason: str | None = "stop"
    logprobs: None = None

class ChatCompletionResponse(BaseModel):
    """POST /v1/chat/completions response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:16]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo

class CompletionRequest(BaseModel):
    """POST /v1/completions request."""
    model: str
    prompt: str | list[str]
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(1, ge=1, le=1)

class CompletionChoice(BaseModel):
    """Single choice in completion response."""
    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    logprobs: dict | None = None

class CompletionResponse(BaseModel):
    """POST /v1/completions response."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:16]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo

class ResponseRequest(BaseModel):
    """POST /v1/responses request."""
    model: str
    input: str | list[Message]
    max_output_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False

class OutputText(BaseModel):
    """Text content block inside ResponseOutput."""
    type: str = "output_text"
    text: str

class ResponseOutput(BaseModel):
    """Output message in /v1/responses."""
    type: str = "message"
    role: str = "assistant"
    content: list[OutputText]

class ResponseObject(BaseModel):
    """POST /v1/responses response."""
    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4().hex[:16]}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: list[ResponseOutput]
    usage: UsageInfo

class EmbeddingRequest(BaseModel):
    """POST /v1/embeddings request."""
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"

class EmbeddingData(BaseModel):
    """Single embedding vector."""
    object: str = "embedding"
    index: int
    embedding: list[float]

class EmbeddingResponse(BaseModel):
    """POST /v1/embeddings response."""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo

class ModelCard(BaseModel):
    """Metadata for a registered model."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"

class ModelListResponse(BaseModel):
    """GET /v1/models response."""
    object: str = "list"
    data: list[ModelCard]
