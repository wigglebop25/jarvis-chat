"""Gemini provider module."""

import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Optional

from ..base import LLMProvider, LLMProviderError, LLMResponse, ToolCall

from .errors import (
    GeminiError,
    GeminiConfigurationError,
    GeminiConnectionError,
    GeminiTimeoutError,
    GeminiRateLimitError,
    GeminiAPIError,
)
from .client import (
    setup_gemini_client,
    get_generation_config,
    get_available_models,
    get_available_models_detailed,
)
from .completion import (
    convert_tools_to_gemini,
    extract_text,
    extract_tool_calls,
    sanitize_gemini_payload,
    setup_context_cache,
    stream_response,
    to_contents,
    usage_from_response,
)
from .request import make_gemini_request

logger = logging.getLogger(__name__)


def _parse_text_action_tool_call(
    text: str,
    tools: list[dict[str, Any]],
) -> ToolCall | None:
    if not text or not tools:
        return None

    clean_text = text.replace("`", "").replace("**", "")
    tool_name_map: dict[str, str] = {}
    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str) and name.strip():
            tool_name_map[name.strip().lower()] = name.strip()

    patterns = [
        r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})?",
        r"\b(?:call|calling|using|executing)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"use the\s+([A-Za-z_][A-Za-z0-9_]*)\s+tool",
    ]

    for pattern in patterns:
        action_match = re.search(pattern, clean_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not action_match:
            continue

        parsed_name = action_match.group(1)
        canonical_name = tool_name_map.get(parsed_name.lower())
        if not canonical_name:
            continue

        args: dict[str, Any] = {}
        inline_args = action_match.group(2) if action_match.lastindex and action_match.lastindex >= 2 else None
        if inline_args:
            try:
                parsed_args = json.loads(inline_args)
                if isinstance(parsed_args, dict):
                    args = parsed_args
            except Exception:
                args = {}
        else:
            args_match = re.search(r"json\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE)
            if args_match:
                try:
                    parsed_args = json.loads(args_match.group(1))
                    if isinstance(parsed_args, dict):
                        args = parsed_args
                except Exception:
                    args = {}

        return ToolCall(id=f"text_{canonical_name}", name=canonical_name, arguments=args)

    return None


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
        
        self._GenerativeModel, self._GenerationConfig, _ = setup_gemini_client(api_key, model)
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature or 0.7
        self.max_tokens = max_tokens or 2048
        # Increased default timeout to 60s for stability with RAG + tool injection
        self.request_timeout_seconds = float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "60"))

        # Configure safety settings to avoid transient 500s from filters
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Initialize the model instance with safety settings
        # We will set system_instruction dynamically during completion if possible
        self.client = self._GenerativeModel(
            model_name=self.model,
            safety_settings=self.safety_settings
        )

        # Check if the model supports native tool calling
        # Gemma models in the Gemini API often don't support native tools yet or are unstable
        self._supports_native_tools = True
        if "gemma" in self.model.lower():
            logger.info(f"Disabling native tools for Gemma-based model: {self.model}")
            self._supports_native_tools = False

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        # Check if we should use native tool calling for this instance
        return getattr(self, "_supports_native_tools", True)

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
                gemini_tools = sanitize_gemini_payload(convert_tools_to_gemini(tools))
            
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
        """
        Generate asynchronous completion using Gemini.
        Includes automatic retry with exponential backoff for rate limits (429).
        """
        try:
            from ...tools.retry_utils import async_retry_with_backoff, RetryExceededError
            return await async_retry_with_backoff(
                lambda: make_gemini_request(self, messages, tools),
                max_retries=3,
                base_delay=2.0,
                retryable_exceptions=(GeminiRateLimitError, GeminiTimeoutError, GeminiError),
            )
        except RetryExceededError as e:
            raise LLMProviderError(f"Gemini request failed after retries: {e}") from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        generation_config = get_generation_config(self.temperature, self.max_tokens)
        
        gemini_tools = None
        if tools and self.supports_tools:
            gemini_tools = sanitize_gemini_payload(convert_tools_to_gemini(tools))

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
