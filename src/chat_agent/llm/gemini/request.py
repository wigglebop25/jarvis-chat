"""
llm/gemini/request.py
──────────────────────
Single Gemini API call execution: system-instruction injection, context
caching for large prompts, and text-fallback tool-call parsing.

Separated from GeminiProvider so it can be tested or swapped independently.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from ..base import LLMProviderError, LLMResponse

from .completion import (
    convert_tools_to_gemini,
    extract_text,
    extract_tool_calls,
    sanitize_gemini_payload,
    setup_context_cache,
    to_contents,
    usage_from_response,
)
from .errors import GeminiError, GeminiRateLimitError, GeminiTimeoutError

if TYPE_CHECKING:
    from . import GeminiProvider

logger = logging.getLogger(__name__)


async def make_gemini_request(
    provider: "GeminiProvider",
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
) -> LLMResponse:
    """
    Execute one Gemini API call.

    Steps:
      1. Extract system instruction (separate field, most stable for Gemini).
      2. Inject tool descriptions as text for models without native tool support.
      3. Optionally enable context caching for prompts ≥ 32K tokens.
      4. Call generate_content_async with timeout.
      5. Run text-fallback parser to catch tool calls in plain-text responses.

    Raises GeminiRateLimitError / GeminiTimeoutError / GeminiError so the
    caller can apply exponential backoff retry logic.
    """
    # Import here to avoid circular reference at module load time
    from . import _parse_text_action_tool_call

    try:
        system_instr = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        filtered_messages = [m for m in messages if m.get("role") != "system"]

        # COMPACT TOOL INJECTION for models without native tool calling
        if tools and not provider.supports_tools:
            tool_desc = "\n\n# TOOLS\n"
            for t in tools:
                tool_desc += f"- {t.get('name')}: {t.get('description', '')}\n"
            system_instr += tool_desc
            system_instr += "\nPROTOCOL:\n1. Execute via: **Action**: tool_name\n"
            system_instr += "2. STOP immediately after the **Action** line.\n"
            system_instr += (
                "3. DO NOT output internal reasoning, planning, or meta-commentary. "
                "Provide ONLY the direct response to the user.\n"
            )

        client = provider._GenerativeModel(
            model_name=provider.model,
            system_instruction=system_instr,
            safety_settings=provider.safety_settings,
        )
        generation_config = provider._GenerationConfig(
            temperature=provider.temperature,
            max_output_tokens=provider.max_tokens,
            stop_sequences=["**Observation**:", "[TOOL_RESULT]", "Thought:", "User says:"],
        )

        gemini_tools = None
        if tools and provider.supports_tools:
            gemini_tools = sanitize_gemini_payload(convert_tools_to_gemini(tools))

        contents = to_contents(filtered_messages)
        cache = None

        # Context caching — only for massive prompts (≥ 32K tokens)
        total_chars = sum(len(m.get("content", "")) for m in filtered_messages)
        if total_chars > 80000:
            total_tokens = sum(provider.count_tokens(m.get("content", "")) for m in filtered_messages)
            if total_tokens >= 32768:
                cache = setup_context_cache(client, provider.model, contents, gemini_tools)
                if cache:
                    client = provider._GenerativeModel.from_cached_content(cached_content=cache)
                    contents = [contents[-1]]

        try:
            kwargs: dict[str, Any] = {
                "contents": contents,
                "generation_config": generation_config,
            }
            if gemini_tools:
                kwargs["tools"] = gemini_tools
            response = await asyncio.wait_for(
                client.generate_content_async(**kwargs),
                timeout=provider.request_timeout_seconds,
            )
        finally:
            if cache:
                try:
                    cache.delete()
                except Exception:
                    pass

        usage = usage_from_response(response)
        text = extract_text(response)
        tool_calls = extract_tool_calls(response) if tools and provider.supports_tools else []

        # Text-fallback parser for models without native tool support
        if not tool_calls and tools:
            if os.getenv("JARVIS_DEBUG_GEMINI_RAW", "0") == "1":
                print(f"\n--- GEMINI RAW RESPONSE ---\n{text}\n---------------------------")
            parsed = _parse_text_action_tool_call(text, tools)
            if parsed:
                logger.info(f"Parsed text tool call fallback: {parsed.name}")
                tool_calls.append(parsed)

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=provider.model,
            usage=usage,
        )

    except asyncio.TimeoutError as e:
        logger.debug(f"Gemini Timeout: timed out after {provider.request_timeout_seconds:.0f}s.")
        raise GeminiTimeoutError(
            f"Gemini request timed out after {provider.request_timeout_seconds:.0f}s"
        ) from e
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "resource_exhausted" in error_msg:
            logger.debug(f"Gemini Rate Limit (429): {e}")
            raise GeminiRateLimitError(f"Gemini rate limit exceeded: {e}") from e
        if any(x in error_msg for x in ("500", "internal error", "503", "service unavailable")):
            logger.debug(f"Gemini Server Error: {e}. Retrying...")
            raise GeminiError(f"Gemini server error: {e}") from e
        logger.error(f"Gemini API Error: {e}")
        raise LLMProviderError(f"Gemini request failed: {e}") from e
