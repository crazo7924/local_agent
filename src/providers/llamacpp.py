"""llama.cpp LLM provider."""

import json
import os
from typing import Any, Callable

from src.providers.base import LLMProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth, ToolCall
from src.providers.utils import callable_to_openai_tool


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
                callable_to_openai_tool(t) if callable(t) else t for t in tools
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
