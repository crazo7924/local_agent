"""Anthropic LLM provider."""

import os
from typing import Any, Callable

from src.providers.base import LLMProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth, ToolCall
from src.providers.utils import callable_to_anthropic_tool


class AnthropicProvider(LLMProvider):
    """LLM provider wrapper for Anthropic / Claude API."""

    def __init__(self, api_key: str | None = None, model_name: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = os.getenv("ANTHROPIC_MODEL", model_name)

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "Anthropic package is not installed. Install optional dependency extra `pip install .[anthropic]`."
            ) from err

        client = anthropic.Anthropic(api_key=self.api_key)

        system_prompt = ""
        formatted_messages = []
        for msg in messages:
            msg_dict = msg.model_dump(exclude_none=True) if isinstance(msg, ChatMessage) else dict(msg)
            role = msg_dict.get("role")

            if role == "system":
                system_prompt += msg_dict.get("content", "") + "\n"
            elif role == "tool":
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": msg_dict.get("tool_call_id", "tool_call_id"),
                                "content": msg_dict.get("content", ""),
                            }
                        ],
                    }
                )
            else:
                formatted_messages.append(
                    {
                        "role": role if role in ("user", "assistant") else "user",
                        "content": msg_dict.get("content", ""),
                    }
                )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": formatted_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            formatted_tools = [
                callable_to_anthropic_tool(t) if callable(t) else t for t in tools
            ]
            kwargs["tools"] = formatted_tools

        response = client.messages.create(**kwargs)

        content_text = ""
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                content_text += block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments=block.input,
                        id=block.id,
                    )
                )

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=response,
        )

    def check_health(self) -> ProviderHealth:
        is_pingable = self.ping_domain("api.anthropic.com")
        if self.api_key:
            return ProviderHealth(
                is_online=True,
                status="Anthropic API configured",
                model=self.model_name,
            )
        return ProviderHealth(
            is_online=is_pingable,
            status="Anthropic API Key not set. Network pingable.",
            model=self.model_name,
        )
