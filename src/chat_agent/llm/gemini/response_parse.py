"""
llm/gemini/response_parse.py
──────────────────────────────
Response extraction, streaming, and context caching for the Gemini API.
  extract_tool_calls   — pull FunctionCall parts from a response
  extract_text         — pull text parts from a response
  usage_from_response  — extract token usage metadata
  stream_response      — async generator for streamed responses
  setup_context_cache  — create a CachedContent for large prompts
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, AsyncGenerator, Optional

from ..base import LLMProviderError, ToolCall
from .message_format import sanitize_gemini_payload, to_contents, to_plain_value

logger = logging.getLogger(__name__)


def extract_tool_calls(response: Any) -> list[ToolCall]:
    """Extract FunctionCall parts from a Gemini response into ToolCall objects."""
    tool_calls: list[ToolCall] = []
    if not hasattr(response, "candidates") or not response.candidates:
        return tool_calls

    call_id = 0
    for candidate in response.candidates:
        if not hasattr(candidate, "content") or not candidate.content:
            continue
        if not hasattr(candidate.content, "parts"):
            continue
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                func_call = part.function_call
                call_id += 1
                arguments: dict[str, Any] = {}
                if hasattr(func_call, "args"):
                    plain_args = to_plain_value(func_call.args)
                    if isinstance(plain_args, dict):
                        arguments = plain_args
                tool_calls.append(
                    ToolCall(
                        id=str(call_id),
                        name=getattr(func_call, "name", ""),
                        arguments=arguments,
                    )
                )
    return tool_calls


def extract_text(response: Any) -> str:
    """Extract and join all text parts from a Gemini response."""
    texts: list[str] = []
    for candidate in (getattr(response, "candidates", None) or []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(part_text)
    return "\n".join(texts).strip()


def usage_from_response(response: Any) -> dict[str, int]:
    """Extract prompt/completion token counts from a Gemini response."""
    usage = getattr(response, "usage_metadata", None)
    usage_dict = usage if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": int(
            getattr(usage, "prompt_token_count", 0)
            or usage_dict.get("prompt_token_count", 0)
            or 0
        ),
        "completion_tokens": int(
            getattr(usage, "candidates_token_count", 0)
            or usage_dict.get("candidates_token_count", 0)
            or usage_dict.get("output_token_count", 0)
            or 0
        ),
    }


async def stream_response(
    client: Any,
    messages: list[dict[str, str]],
    generation_config: Any,
    tools: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """Stream text chunks from a Gemini generate_content_async call."""
    try:
        safe_tools = sanitize_gemini_payload(tools) if tools is not None else None
        streamed = await client.generate_content_async(
            to_contents(messages),
            stream=True,
            generation_config=generation_config,
            tools=safe_tools,
        )
        async for chunk in streamed:
            text = extract_text(chunk)
            if text:
                yield text
    except Exception as e:
        raise LLMProviderError(f"Gemini streaming failed: {e}") from e


def setup_context_cache(
    client: Any,
    model: str,
    contents: list[dict[str, Any]],
    gemini_tools: Optional[dict[str, Any]] = None,
    ttl_minutes: int = 5,
) -> Optional[Any]:
    """
    Create a Gemini CachedContent for prompts that exceed 32K tokens.
    Returns the cache object or None if caching is unavailable.
    """
    try:
        safe_tools = sanitize_gemini_payload(gemini_tools) if gemini_tools is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from google.generativeai import caching
        import datetime

        cache = caching.CachedContent.create(
            model=model,
            contents=contents[:-1],
            tools=safe_tools,
            ttl=datetime.timedelta(minutes=ttl_minutes),
        )
        return cache
    except Exception as e:
        logger.debug(f"Context caching not available: {e}")
        return None
