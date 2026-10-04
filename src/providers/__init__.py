"""LLM Provider package imports and factory functions."""

from typing import Type

from src.providers.anthropic import AnthropicProvider
from src.providers.base import LLMProvider
from src.providers.dummy import DummyProvider, MockProvider
from src.providers.gemini import GeminiProvider
from src.providers.llamacpp import LlamaCppProvider
from src.providers.ollama import OllamaProvider
from src.providers.openai import OpenAIProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth, ToolCall

PROVIDERS: dict[str, Type[LLMProvider]] = {
    "dummy": DummyProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "llamacpp": LlamaCppProvider,
    "llama.cpp": LlamaCppProvider,
}


def get_provider(name: str = "dummy", **kwargs) -> LLMProvider:
    """Factory function to instantiate an LLM provider by name."""
    if name.lower() == "mock":
        return MockProvider(**kwargs)

    provider_cls = PROVIDERS.get(name.lower())
    if not provider_cls:
        raise ValueError(
            f"Unknown provider '{name}'. Available options: {list(PROVIDERS.keys())}"
        )
    return provider_cls(**kwargs)


__all__ = [
    "LLMProvider",
    "DummyProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "LlamaCppProvider",
    "ToolCall",
    "ChatMessage",
    "LLMResponse",
    "ProviderHealth",
    "get_provider",
    "PROVIDERS",
]
