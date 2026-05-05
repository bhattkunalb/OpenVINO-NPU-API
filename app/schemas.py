"""OpenAI-compatible Pydantic request/response schemas."""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single chat message with a role and text content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class UsageInfo(BaseModel):
    """Token usage counters returned with every completion response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorDetail(BaseModel):
    """Inner error object matching the OpenAI error envelope."""

    message: str
    type: str = "server_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Top-level OpenAI-compatible error envelope."""

    error: ErrorDetail


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    """Request schema for /v1/chat/completions."""
    """Request body for POST /v1/chat/completions."""

    model: str
    messages: list[Message]
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    n: int = Field(1, ge=1, le=1)  # NPU runs single inference only

    @field_validator("n")
    @classmethod
    def single_completion_only(cls, v: int) -> int:
        """Reject n>1; NPU pipeline runs one inference at a time."""
        if v != 1:
            raise ValueError("n>1 not supported; NPU runs single inference only.")
        return v


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""
    """A single completion choice in a non-streaming response."""

    index: int = 0
    message: Message
    finish_reason: Optional[str] = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    """Response schema for /v1/chat/completions."""
    """Full response envelope for POST /v1/chat/completions (non-streaming)."""

    # "chatcmpl-" is the standard OpenAI prefix for chat completion IDs
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# SSE streaming chunk shapes

class DeltaContent(BaseModel):
    """Incremental content delta inside a streaming chunk."""

    content: Optional[str] = None
    role: Optional[str] = None


class StreamChoice(BaseModel):
    """A single choice entry inside a streaming SSE chunk."""

    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming chat completion response."""
    """SSE chunk sent during a streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ---------------------------------------------------------------------------
# /v1/completions (Legacy/Standard Completions)
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    """Request schema for /v1/completions (legacy text completions)."""
    """Request body for POST /v1/completions."""

    model: str
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    n: int = Field(1, ge=1, le=1)

    @field_validator("n")
    @classmethod
    def single_completion_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError("n>1 not supported; NPU runs single inference only.")
        return v


class CompletionChoice(BaseModel):
    """A single choice in a non-streaming completion response."""

    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[dict] = None


class CompletionResponse(BaseModel):
    """Response schema for /v1/completions."""
    """Full response envelope for POST /v1/completions."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# /v1/responses  (OpenAI Responses API style)
# ---------------------------------------------------------------------------

class ResponseRequest(BaseModel):
    """Request body for POST /v1/responses."""

    model: str
    input: Union[str, list[Message]]
    max_output_tokens: Optional[int] = Field(None, ge=1, le=32768)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False


class OutputText(BaseModel):
    """A text content block inside a ResponseOutput."""

    type: str = "output_text"
    text: str


class ResponseOutput(BaseModel):
    """A single output message in a /v1/responses reply."""

    type: str = "message"
    role: str = "assistant"
    content: list[OutputText]


class ResponseObject(BaseModel):
    """Top-level response envelope for POST /v1/responses."""

    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4().hex}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: list[ResponseOutput]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    """Request schema for /v1/embeddings."""
    """Request body for POST /v1/embeddings."""

    model: str
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    """A single embedding vector with its index."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Response schema for /v1/embeddings."""
    """Response envelope for POST /v1/embeddings."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

class ModelCard(BaseModel):
    """Metadata card for a single registered model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Response envelope for GET /v1/models."""

    object: str = "list"
    data: list[ModelCard]
