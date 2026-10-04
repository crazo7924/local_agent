"""Dummy and Mock LLM providers."""

from typing import Any, Callable

from src.providers.base import LLMProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth


class DummyProvider(LLMProvider):
    """Default provider that raises an exception reminding user to configure an LLM provider explicitly."""

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        raise RuntimeError(
            "No active LLM provider configured! "
            "Please explicitly configure a provider (e.g., OllamaProvider, OpenAIProvider, "
            "AnthropicProvider, GeminiProvider, LlamaCppProvider, or MockProvider)."
        )

    def check_health(self) -> ProviderHealth:
        return ProviderHealth(
            is_online=False,
            status="DummyProvider is active: Unconfigured provider. Explicit configuration required.",
            model="None",
        )


class MockProvider(LLMProvider):
    """Mock LLM provider designed for testing and offline execution."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        default_content: str = "Mock response",
        model_name: str = "mock-model",
    ):
        self.responses = responses or []
        self.default_content = default_content
        self.model_name = model_name
        self.call_history: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        self.call_history.append({"messages": messages, "tools": tools})

        if self.responses:
            return self.responses.pop(0)

        return LLMResponse(content=self.default_content)

    def check_health(self) -> ProviderHealth:
        return ProviderHealth(
            is_online=True,
            status="MockProvider is online",
            model=self.model_name,
        )
