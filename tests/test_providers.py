"""Tests for LLM providers and factory."""

import pytest
from src.providers import (
    AnthropicProvider,
    DummyProvider,
    GeminiProvider,
    LlamaCppProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)
from src.providers.types import LLMResponse, ToolCall


def test_dummy_provider_raises_error():
    provider = DummyProvider()
    with pytest.raises(RuntimeError, match="No active LLM provider configured!"):
        provider.chat([{"role": "user", "content": "Hello"}])

    health = provider.check_health()
    assert health.is_online is False
    assert "DummyProvider is active" in health.status


def test_mock_provider_responses():
    response1 = LLMResponse(content="First response")
    response2 = LLMResponse(
        tool_calls=[ToolCall(name="list_directory", arguments={"directory_path": "."})]
    )
    provider = MockProvider(responses=[response1, response2])

    res1 = provider.chat([{"role": "user", "content": "Hi"}])
    assert res1.content == "First response"

    res2 = provider.chat([{"role": "user", "content": "List files"}])
    assert res2.tool_calls[0].name == "list_directory"

    res3 = provider.chat([{"role": "user", "content": "Default"}])
    assert res3.content == "Mock response"

    health = provider.check_health()
    assert health.is_online is True


def test_get_provider_factory():
    assert isinstance(get_provider("dummy"), DummyProvider)
    assert isinstance(get_provider("mock"), MockProvider)
    assert isinstance(get_provider("ollama"), OllamaProvider)
    assert isinstance(get_provider("openai"), OpenAIProvider)
    assert isinstance(get_provider("anthropic"), AnthropicProvider)
    assert isinstance(get_provider("gemini"), GeminiProvider)
    assert isinstance(get_provider("llamacpp"), LlamaCppProvider)

    with pytest.raises(ValueError, match="Unknown provider 'invalid'"):
        get_provider("invalid")
