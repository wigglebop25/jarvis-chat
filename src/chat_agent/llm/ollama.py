import json
from typing import Any, AsyncGenerator, Optional

import httpx

from .base import LLMProvider, LLMProviderError, LLMResponse


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local model serving."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supports_tools(self) -> bool:
        return False

    def is_configured(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate synchronous completion using Ollama."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }

            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )
        except httpx.HTTPError as e:
            raise LLMProviderError(f"Ollama request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMProviderError(f"Invalid response from Ollama: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate asynchronous completion using Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                }

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                return LLMResponse(
                    text=data.get("message", {}).get("content", ""),
                    model=self.model,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    },
                )
            except httpx.HTTPError as e:
                raise LLMProviderError(f"Ollama request failed: {e}") from e
            except json.JSONDecodeError as e:
                raise LLMProviderError(f"Invalid response from Ollama: {e}") from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }

                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
            except httpx.HTTPError as e:
                raise LLMProviderError(f"Ollama streaming failed: {e}") from e

    def __del__(self):
        self.client.close()
