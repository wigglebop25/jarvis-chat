from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class LLMConfigurationError(LLMProviderError):
    """Raised when LLM provider is not properly configured."""


@dataclass
class ToolCall:
    """Represents a tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'ollama', 'openai', 'gemini')."""
        ...

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports function calling."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider has required configuration."""
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        ...

    def get_available_models(self) -> list[str]:
        """Fetch available models from the provider's API."""
        return [getattr(self, "model", "")]

    @abstractmethod
    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Synchronous completion. Messages format: [{'role': 'user'/'assistant', 'content': '...'}]"""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Asynchronous completion."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response text. Yields text chunks."""
        ...
