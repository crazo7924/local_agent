"""Pydantic models for LLM provider types and interfaces."""

from typing import Any
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Standard representation of an LLM tool call request."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class ChatMessage(BaseModel):
    """Standard chat message format."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class LLMResponse(BaseModel):
    """Standard response returned by LLM providers."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw_response: Any = None


class ProviderHealth(BaseModel):
    """Health check status representation for LLM providers."""

    is_online: bool
    status: str
    model: str | None = None
    details: dict[str, Any] | None = None
