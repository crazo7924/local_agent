"""Tests for agent loop execution with dependency injection."""

import pytest
from src.agent import run_agent_loop
from src.providers import MockProvider
from src.providers.types import LLMResponse, ToolCall


def test_agent_loop_with_dummy_provider():
    with pytest.raises(RuntimeError, match="No active LLM provider configured!"):
        run_agent_loop("Hello")


def test_agent_loop_with_mock_provider_direct_answer():
    provider = MockProvider(default_content="Hello there! How can I help you?")
    result = run_agent_loop("Hello", provider=provider)
    assert result == "Hello there! How can I help you?"


def test_agent_loop_with_mock_provider_tool_call():
    # Sequence of responses:
    # 1. Ask agent to list directory
    # 2. Return final response after tool call
    res_tool_call = LLMResponse(
        tool_calls=[ToolCall(name="list_directory", arguments={"directory_path": "."})]
    )
    res_final = LLMResponse(content="Found files in directory.")

    provider = MockProvider(responses=[res_tool_call, res_final])

    result = run_agent_loop("List current directory", provider=provider)
    assert result == "Found files in directory."
    assert len(provider.call_history) == 2
