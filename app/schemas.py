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
    """Detailed error information including message and code."""

    message: str
    type: str = "server_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response envelope following OpenAI convention."""

    error: ErrorDetail


class ChatCompletionRequest(BaseModel):
    """Request schema for the chat completions endpoint."""

    model: str
    messages: list[Message]
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    n: int = Field(1, ge=1, le=1)


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: Message
    finish_reason: str | None = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    """Full response schema for the chat completions endpoint."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:16]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: str = "ov_genai_npu_v1"


class CompletionRequest(BaseModel):
    """Request schema for the legacy text completions endpoint."""

    model: str
    prompt: str | list[str]
    max_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None


class CompletionChoice(BaseModel):
    """A single choice in a text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    logprobs: dict | None = None


class CompletionResponse(BaseModel):
    """Response schema for the text completion endpoint."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:16]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ResponseRequest(BaseModel):
    """Request schema for the internal /v1/responses endpoint."""

    model: str
    input: str | list[Message]
    max_output_tokens: int | None = Field(None, ge=1, le=32768)
    temperature: float | None = Field(1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(1.0, ge=0.0, le=1.0)
    stream: bool = False


class OutputText(BaseModel):
    """A single text chunk in a multi-modal response output."""

    type: str = "output_text"
    text: str


class ResponseOutput(BaseModel):
    """Unified output container for the /v1/responses endpoint."""

    type: str = "message"
    role: str = "assistant"
    content: list[OutputText]


class ResponseObject(BaseModel):
    """Response object for the /v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4().hex[:16]}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: list[ResponseOutput]
    usage: UsageInfo


class EmbeddingRequest(BaseModel):
    """Request schema for the embeddings endpoint."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    """A single embedding vector with its position index."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Response schema for the embeddings endpoint."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


class ModelCard(BaseModel):
    """OpenAI-compatible model descriptor."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ov_genai"


class ModelListResponse(BaseModel):
    """Response schema for the model list endpoint."""

    object: str = "list"
    data: list[ModelCard]


# Project: OpenVINO NPU Inference API
# Version: 1.0.0
# Stability: Production
