"""Gemini completion and streaming operations."""

import logging
import warnings
from typing import Any, AsyncGenerator, Optional

from ..base import ToolCall, LLMProviderError

logger = logging.getLogger(__name__)


def to_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Convert messages to Gemini contents format.
    
    Args:
        messages: List of message dicts with role and content
        
    Returns:
        List of content dicts for Gemini API
    """
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        # Map system to user for Gemini
        mapped_role = "user" if role in {"user", "system"} else "model"
        
        content_text = msg.get("content", "")
        if not content_text and role != "tool":
            continue

        # Merge consecutive messages with the same role
        if contents and contents[-1]["role"] == mapped_role:
            contents[-1]["parts"][0]["text"] += "\n\n" + content_text
        else:
            contents.append(
                {
                    "role": mapped_role,
                    "parts": [{"text": content_text}],
                }
            )
    return contents


def convert_tools_to_gemini(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Convert tool definitions to Gemini format.
    
    Args:
        tools: List of tool definitions
        
    Returns:
        Gemini tools object with function_declarations
    """
    try:
        from ..tools.schemas import ToolSchemaConverter  # type: ignore
    except Exception:
        ToolSchemaConverter = None

    function_declarations = []
    for tool in tools:
        if ToolSchemaConverter is not None:
            try:
                function_declarations.append(ToolSchemaConverter.to_gemini(tool))
                continue
            except Exception:
                pass
        # Fallback: construct a minimal function declaration
        params = tool.get("parameters") or tool.get("inputSchema") or {}
        function_declarations.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": params,
        })

    return {"function_declarations": function_declarations}


def to_plain_value(value: Any) -> Any:
    """
    Convert protobuf/map/repeated values to plain Python JSON-safe values.
    
    Args:
        value: Value to convert
        
    Returns:
        Plain Python value
    """
    if isinstance(value, dict):
        return {str(k): to_plain_value(v) for k, v in value.items()}

    if hasattr(value, "ListFields"):
        return {
            getattr(field, "name", str(field)): to_plain_value(field_value)
            for field, field_value in value.ListFields()
        }

    if hasattr(value, "items") and not isinstance(value, (str, bytes)):
        try:
            return {str(k): to_plain_value(v) for k, v in value.items()}
        except TypeError:
            pass

    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [to_plain_value(item) for item in value]
        except TypeError:
            pass

    return value


def extract_tool_calls(response: Any) -> list[ToolCall]:
    """
    Extract tool calls from Gemini response.
    
    Args:
        response: Response from Gemini API
        
    Returns:
        List of ToolCall objects
    """
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
                
                arguments = {}
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
    """
    Extract text from Gemini response.
    
    Args:
        response: Response from Gemini API
        
    Returns:
        Extracted text content
    """
    texts: list[str] = []

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(part_text)

    return "\n".join(texts).strip()


def usage_from_response(response: Any) -> dict[str, int]:
    """
    Extract usage metrics from response.
    
    Args:
        response: Response from Gemini API
        
    Returns:
        Dict with prompt_tokens and completion_tokens
    """
    usage = getattr(response, "usage_metadata", None)
    usage_dict = usage if isinstance(usage, dict) else {}

    prompt_tokens = int(
        getattr(usage, "prompt_token_count", 0)
        or usage_dict.get("prompt_token_count", 0)
        or 0
    )
    completion_tokens = int(
        getattr(usage, "candidates_token_count", 0)
        or usage_dict.get("candidates_token_count", 0)
        or usage_dict.get("output_token_count", 0)
        or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def stream_response(
    client: Any,
    messages: list[dict[str, str]],
    generation_config: Any,
    tools: Optional[dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream response from Gemini.
    
    Args:
        client: GenerativeModel client instance
        messages: List of messages
        generation_config: GenerationConfig instance
        tools: Optional tools dict
        
    Yields:
        Text chunks from the response
        
    Raises:
        LLMProviderError: If streaming fails
    """
    try:
        stream_response = await client.generate_content_async(
            to_contents(messages),
            stream=True,
            generation_config=generation_config,
            tools=tools,
        )
        async for chunk in stream_response:
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
    Set up context caching for large prompts.
    
    Args:
        client: GenerativeModel client
        model: Model name
        contents: Message contents
        gemini_tools: Optional tools dict
        ttl_minutes: Cache TTL in minutes
        
    Returns:
        CachedContent instance or None if caching not available
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from google.generativeai import caching
        import datetime
        
        cache = caching.CachedContent.create(
            model=model,
            contents=contents[:-1],
            tools=gemini_tools,
            ttl=datetime.timedelta(minutes=ttl_minutes),
        )
        return cache
    except Exception as e:
        logger.debug(f"Context caching not available: {e}")
        return None
