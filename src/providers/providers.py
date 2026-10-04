"""Implementations of LLM providers."""

import inspect
import json
import os
from typing import Any, Callable

from src.providers.base import LLMProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth, ToolCall


def _callable_to_openai_tool(func: Callable) -> dict[str, Any]:
    """Converts a Python function to an OpenAI/llama.cpp tool definition schema."""
    doc = inspect.getdoc(func) or ""
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is float:
            param_type = "number"
        elif param.annotation is bool:
            param_type = "boolean"
        elif param.annotation in (dict, list):
            param_type = "object"

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _callable_to_anthropic_tool(func: Callable) -> dict[str, Any]:
    """Converts a Python function to an Anthropic tool definition schema."""
    doc = inspect.getdoc(func) or ""
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is float:
            param_type = "number"
        elif param.annotation is bool:
            param_type = "boolean"

        properties[param_name] = {
            "type": param_type,
            "description": f"Parameter {param_name}",
        }
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": func.__name__,
        "description": doc.strip(),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


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


class OpenAIProvider(LLMProvider):
    """LLM provider wrapper for OpenAI API."""

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", model_name)

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "OpenAI package is not installed. Install optional dependency extra `pip install .[openai]`."
            ) from err

        client = openai.OpenAI(api_key=self.api_key)

        formatted_messages = []
        for msg in messages:
            msg_dict = msg.model_dump(exclude_none=True) if isinstance(msg, ChatMessage) else dict(msg)

            # Format tool call responses according to OpenAI chat completions spec
            if msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                formatted_tc = []
                for tc in msg_dict["tool_calls"]:
                    formatted_tc.append(
                        {
                            "id": tc.get("id") or f"call_{tc.get('name')}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                    )
                msg_dict["tool_calls"] = formatted_tc

            formatted_messages.append(msg_dict)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        if tools:
            formatted_tools = [
                _callable_to_openai_tool(t) if callable(t) else t for t in tools
            ]
            kwargs["tools"] = formatted_tools

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = None
        if choice.tool_calls:
            tool_calls = []
            for tc in choice.tool_calls:
                args = (
                    json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments
                )
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=args,
                        id=tc.id,
                    )
                )

        return LLMResponse(
            content=choice.content,
            tool_calls=tool_calls,
            raw_response=response,
        )

    def check_health(self) -> ProviderHealth:
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            client.models.retrieve(self.model_name)
            return ProviderHealth(
                is_online=True,
                status="OpenAI API is online",
                model=self.model_name,
            )
        except Exception as e:
            is_pingable = self.ping_domain("api.openai.com")
            return ProviderHealth(
                is_online=is_pingable,
                status=f"OpenAI health check notice: {str(e)}",
                model=self.model_name,
            )


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
                _callable_to_anthropic_tool(t) if callable(t) else t for t in tools
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


class GeminiProvider(LLMProvider):
    """LLM provider wrapper for Google Gemini API."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", model_name)

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        try:
            from google import genai
        except ImportError as err:
            raise ImportError(
                "google-genai package is not installed. Install optional dependency extra `pip install .[gemini]`."
            ) from err

        client = genai.Client(api_key=self.api_key)

        prompt_parts = []
        for msg in messages:
            msg_dict = msg.model_dump(exclude_none=True) if isinstance(msg, ChatMessage) else dict(msg)
            role = msg_dict.get("role", "user")
            content = msg_dict.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        full_prompt = "\n".join(prompt_parts)

        config = {}
        if tools:
            config["tools"] = tools

        response = client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=config if config else None,
        )

        tool_calls = []
        if getattr(response, "function_calls", None):
            for fc in response.function_calls:
                tool_calls.append(
                    ToolCall(
                        name=fc.name,
                        arguments=dict(fc.args),
                    )
                )

        return LLMResponse(
            content=response.text if hasattr(response, "text") else None,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=response,
        )

    def check_health(self) -> ProviderHealth:
        is_pingable = self.ping_domain("generativelanguage.googleapis.com")
        if self.api_key:
            return ProviderHealth(
                is_online=True,
                status="Gemini API configured",
                model=self.model_name,
            )
        return ProviderHealth(
            is_online=is_pingable,
            status="Gemini API Key not set. Network pingable.",
            model=self.model_name,
        )


class LlamaCppProvider(LLMProvider):
    """LLM provider wrapper for llama.cpp standard OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model_name: str = "llama-cpp",
        api_key: str = "no-key-required",
    ):
        self.base_url = os.getenv("LLAMACPP_BASE_URL", base_url)
        self.model_name = os.getenv("LLAMACPP_MODEL", model_name)
        self.api_key = api_key

    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "OpenAI package is required for LlamaCppProvider. Install with `pip install openai`."
            ) from err

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

        formatted_messages = []
        for msg in messages:
            msg_dict = msg.model_dump(exclude_none=True) if isinstance(msg, ChatMessage) else dict(msg)

            if msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                formatted_tc = []
                for tc in msg_dict["tool_calls"]:
                    formatted_tc.append(
                        {
                            "id": tc.get("id") or f"call_{tc.get('name')}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                    )
                msg_dict["tool_calls"] = formatted_tc

            formatted_messages.append(msg_dict)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        if tools:
            formatted_tools = [
                _callable_to_openai_tool(t) if callable(t) else t for t in tools
            ]
            kwargs["tools"] = formatted_tools

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = None
        if choice.tool_calls:
            tool_calls = []
            for tc in choice.tool_calls:
                args = (
                    json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments
                )
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=args,
                        id=tc.id,
                    )
                )

        return LLMResponse(
            content=choice.content,
            tool_calls=tool_calls,
            raw_response=response,
        )

    def check_health(self) -> ProviderHealth:
        try:
            import urllib.request

            req = urllib.request.Request(f"{self.base_url.rstrip('/')}/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return ProviderHealth(
                        is_online=True,
                        status="llama.cpp server is online",
                        model=self.model_name,
                    )
        except Exception:
            pass

        return ProviderHealth(
            is_online=False,
            status="llama.cpp server unreachable at " + self.base_url,
            model=self.model_name,
        )
