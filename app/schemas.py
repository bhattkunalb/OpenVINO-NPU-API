"""OpenAI-compatible Pydantic request/response schemas."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message with a role and content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class UsageInfo(BaseModel):
    """Token usage counters for a request/response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorDetail(BaseModel):
    """Details about a specific error."""

    message: str
    type: str = "server_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error: ErrorDetail


class ChatCompletionRequest(BaseModel):
    """Parameters for creating a chat completion."""

    model: str
    messages: list[Message]
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(1, ge=1, le=1)


class ChatCompletionChoice(BaseModel):
    """A single result choice for a chat completion."""

    index: int = 0
    message: Message
    finish_reason: str | None = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    """A full response object for a chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:16]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    """Parameters for creating a completion."""

    model: str
    prompt: str | list[str]
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(1, ge=1, le=1)


class CompletionChoice(BaseModel):
    """A single result choice for a completion."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    logprobs: dict | None = None


class CompletionResponse(BaseModel):
    """A full response object for a completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:16]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ResponseRequest(BaseModel):
    """Parameters for creating a response object."""

    model: str
    input: str | list[Message]
    max_output_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False


class OutputText(BaseModel):
    """A text segment of the response output."""

    type: str = "output_text"
    text: str


class ResponseOutput(BaseModel):
    """A single output object in the response list."""

    type: str = "message"
    role: str = "assistant"
    content: list[OutputText]


class ResponseObject(BaseModel):
    """A full response object for the /v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4().hex[:16]}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: list[ResponseOutput]
    usage: UsageInfo


class EmbeddingRequest(BaseModel):
    """Parameters for creating an embedding."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    """A single embedding vector object."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """A full response object for the /v1/embeddings endpoint."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


class ModelCard(BaseModel):
    """Metadata for a registered OpenVINO model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """A list of available models."""

    object: str = "list"
    data: list[ModelCard]

