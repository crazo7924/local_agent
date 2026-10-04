"""Ollama LLM provider."""

import os
from typing import Any, Callable

from src.providers.base import LLMProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth, ToolCall


class OllamaProvider(LLMProvider):
    """LLM provider wrapper for local Ollama instance."""

    def __init__(self, model_name: str = "llama3.2:latest", host: str | None = None):
        self.model_name = os.getenv("OLLAMA_MODEL", model_name)
        self.host = host or os.getenv("OLLAMA_HOST")

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        try:
            import ollama
        except ImportError as err:
            raise ImportError(
                "Ollama package is not installed. Install with `pip install ollama`."
            ) from err

        if self.host:
            client = ollama.Client(host=self.host)
        else:
            client = ollama

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append(msg.model_dump(exclude_none=True))
            else:
                formatted_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = client.chat(**kwargs)
        msg_obj = response.get("message", {})

        tool_calls = None
        if msg_obj.get("tool_calls"):
            tool_calls = []
            for tc in msg_obj["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                        id=tc.get("id"),
                    )
                )

        return LLMResponse(
            content=msg_obj.get("content"),
            tool_calls=tool_calls,
            raw_response=response,
        )

    def check_health(self) -> ProviderHealth:
        try:
            import ollama

            if self.host:
                client = ollama.Client(host=self.host)
            else:
                client = ollama

            ps_response = client.ps()
            models = getattr(ps_response, "models", [])
            for model in models:
                if model.name == self.model_name:
                    return ProviderHealth(
                        is_online=True,
                        status="Agent is online",
                        model=self.model_name,
                    )

            return ProviderHealth(
                is_online=True,
                status=f"Ollama server is reachable. Model '{self.model_name}' ready.",
                model=self.model_name,
            )
        except Exception as e:
            is_pingable = False
            if self.host:
                is_pingable = self.ping_domain(self.host)

            return ProviderHealth(
                is_online=is_pingable,
                status=f"Ollama unreachable: {str(e)}",
                model=self.model_name,
            )
