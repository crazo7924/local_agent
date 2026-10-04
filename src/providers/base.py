"""Abstract base class for LLM providers."""

import subprocess
from abc import ABC, abstractmethod
from typing import Any, Callable

from src.providers.types import ChatMessage, LLMResponse, ProviderHealth


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[Callable] | None = None,
    ) -> LLMResponse:
        """Send chat messages and tool definitions to the LLM and receive a standardized response."""
        pass

    @abstractmethod
    def check_health(self) -> ProviderHealth:
        """Check the health and availability of the LLM provider."""
        pass

    def ping_domain(self, host: str, timeout_seconds: int = 2) -> bool:
        """Helper method to check host reachability via ping or network probe."""
        try:
            # Use -c 1 for 1 packet, -W timeout in seconds on Linux
            res = subprocess.run(
                ["ping", "-c", "1", "-W", str(timeout_seconds), host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return res.returncode == 0
        except Exception:
            return False
