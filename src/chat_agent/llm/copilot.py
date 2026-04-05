from typing import Any, AsyncGenerator, Optional

from .base import LLMProvider, LLMResponse


class CopilotProvider(LLMProvider):
    """GitHub Copilot provider using github-copilot-sdk library."""

    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model

    @property
    def name(self) -> str:
        return "copilot"

    @property
    def supports_tools(self) -> bool:
        return True

    def is_configured(self) -> bool:
        raise NotImplementedError(
            "Copilot provider requires GitHub CLI authentication. "
            "Authenticate with 'gh auth login' and ensure you have access to GitHub Copilot."
        )

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Synchronous completion is not yet implemented for Copilot provider."""
        raise NotImplementedError(
            "Copilot provider support is not yet implemented. "
            "Use OpenAI, Gemini, or Ollama providers instead."
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Asynchronous completion is not yet implemented for Copilot provider."""
        raise NotImplementedError(
            "Copilot provider support is not yet implemented. "
            "Use OpenAI, Gemini, or Ollama providers instead."
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming is not yet implemented for Copilot provider."""
        raise NotImplementedError(
            "Copilot provider support is not yet implemented. "
            "Use OpenAI, Gemini, or Ollama providers instead."
        )
        yield
