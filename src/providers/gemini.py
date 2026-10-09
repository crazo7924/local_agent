"""Gemini LLM provider."""

import os
from typing import Any, Callable

from src.providers.base import LLMProvider
from src.providers.types import ChatMessage, LLMResponse, ProviderHealth, ToolCall


class GeminiProvider(LLMProvider):
    """LLM provider wrapper for Google Gemini API."""

    def __init__(
        self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"
    ):
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
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
            msg_dict = (
                msg.model_dump(exclude_none=True)
                if isinstance(msg, ChatMessage)
                else dict(msg)
            )
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
