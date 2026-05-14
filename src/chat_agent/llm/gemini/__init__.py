"""Gemini provider module."""

import os
import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

from ..base import LLMProvider, LLMProviderError, LLMResponse

from .errors import (
    GeminiError,
    GeminiConfigurationError,
    GeminiConnectionError,
    GeminiTimeoutError,
    GeminiRateLimitError,
    GeminiAPIError,
)
from .retry import retry_with_backoff, MaxRetriesExceeded
from .client import (
    setup_gemini_client,
    get_generation_config,
    get_available_models,
    get_available_models_detailed,
)
from .completion import (
    to_contents,
    convert_tools_to_gemini,
    extract_tool_calls,
    extract_text,
    usage_from_response,
    stream_response,
    setup_context_cache,
)

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini / AI Studio API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY or GOOGLE_API_KEY must be set to initialize Gemini provider")
        
        self._GenerativeModel, self._GenerationConfig, self.client = setup_gemini_client(api_key, model)
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature or 0.7
        self.max_tokens = max_tokens or 2048
        self.request_timeout_seconds = float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "30"))

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        return True

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_available_models(self) -> list[str]:
        models = get_available_models()
        return models if models else [self.model]

    def get_available_models_detailed(self) -> list[dict[str, Any]]:
        models = get_available_models_detailed()
        return models if models else [{"name": self.model, "input_token_limit": 0}]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return self.client.count_tokens(text).total_tokens
        except Exception:
            return len(text) // 4

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            generation_config = get_generation_config(self.temperature, self.max_tokens)
            
            gemini_tools = None
            if tools and self.supports_tools:
                gemini_tools = convert_tools_to_gemini(tools)
            
            contents = to_contents(messages)
            client = self.client
            cache = None
            
            # Context Caching for large prompts
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > 80000 and len(contents) > 1:
                total_tokens = sum(self.count_tokens(m.get("content", "")) for m in messages)
                if total_tokens >= 32768:
                    cache = setup_context_cache(client, self.model, contents, gemini_tools)
                    if cache:
                        client = self._GenerativeModel.from_cached_content(cached_content=cache)
                        contents = [contents[-1]]

            try:
                if gemini_tools:
                    response = client.generate_content(contents, generation_config=generation_config, tools=gemini_tools, stream=False)
                else:
                    response = client.generate_content(contents, generation_config=generation_config, stream=False)
            finally:
                if cache:
                    try:
                        cache.delete()
                    except Exception:
                        pass

            usage = usage_from_response(response)
            logger.info(f"Gemini Usage: {usage.get('prompt_tokens', 0)} prompt tokens, {usage.get('completion_tokens', 0)} completion tokens")
            
            tool_calls = extract_tool_calls(response) if tools else []
            
            return LLMResponse(
                text=extract_text(response),
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
            )
        except Exception as e:
            import traceback
            logger.error(f"Gemini API Error: {e}\n{traceback.format_exc()}")
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            generation_config = get_generation_config(self.temperature, self.max_tokens)
            
            gemini_tools = None
            if tools and self.supports_tools:
                gemini_tools = convert_tools_to_gemini(tools)
            
            contents = to_contents(messages)
            client = self.client
            cache = None
            
            # Context Caching for large prompts
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > 80000 and len(contents) > 1:
                total_tokens = sum(self.count_tokens(m.get("content", "")) for m in messages)
                if total_tokens >= 32768:
                    cache = setup_context_cache(client, self.model, contents, gemini_tools)
                    if cache:
                        client = self._GenerativeModel.from_cached_content(cached_content=cache)
                        contents = [contents[-1]]

            try:
                response = await asyncio.wait_for(
                    client.generate_content_async(
                        contents,
                        generation_config=generation_config,
                        tools=gemini_tools,
                    ),
                    timeout=self.request_timeout_seconds,
                )
            finally:
                if cache:
                    try:
                        cache.delete()
                    except Exception:
                        pass
            
            usage = usage_from_response(response)
            logger.info(f"Gemini Usage: {usage.get('prompt_tokens', 0)} prompt tokens, {usage.get('completion_tokens', 0)} completion tokens")

            tool_calls = extract_tool_calls(response) if tools else []
            
            return LLMResponse(
                text=extract_text(response),
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
            )
        except asyncio.TimeoutError as e:
            logger.error(f"Gemini Timeout: Request timed out after {self.request_timeout_seconds:.0f}s. This can happen with very large prompts or high traffic.")
            raise LLMProviderError(
                f"Gemini request timed out after {self.request_timeout_seconds:.0f}s"
            ) from e
        except Exception as e:
            import traceback
            error_msg = str(e)
            if "404" in error_msg:
                logger.error(f"Gemini API Error: Model '{self.model}' not found or unsupported. Please use a valid model like 'gemini-1.5-flash'.")
            else:
                logger.error(f"Gemini API Error: {e}\n{traceback.format_exc()}")
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        generation_config = get_generation_config(self.temperature, self.max_tokens)
        
        gemini_tools = None
        if tools and self.supports_tools:
            gemini_tools = convert_tools_to_gemini(tools)

        async for chunk in stream_response(self.client, messages, generation_config, gemini_tools):
            yield chunk


__all__ = [
    "GeminiProvider",
    "GeminiError",
    "GeminiConfigurationError",
    "GeminiConnectionError",
    "GeminiTimeoutError",
    "GeminiRateLimitError",
    "GeminiAPIError",
    "retry_with_backoff",
    "MaxRetriesExceeded",
    "setup_gemini_client",
    "get_generation_config",
    "get_available_models",
    "get_available_models_detailed",
    "to_contents",
    "convert_tools_to_gemini",
    "extract_tool_calls",
    "extract_text",
    "usage_from_response",
    "stream_response",
    "setup_context_cache",
]
