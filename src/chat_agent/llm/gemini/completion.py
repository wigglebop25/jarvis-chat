"""
llm/gemini/completion.py
─────────────────────────
Backward-compatible re-export. Implementation split into:
  - message_format.py  — to_contents, sanitize_gemini_payload,
                         to_plain_value, convert_tools_to_gemini
  - response_parse.py  — extract_tool_calls, extract_text,
                         usage_from_response, stream_response,
                         setup_context_cache
"""
from .message_format import (
    convert_tools_to_gemini,
    sanitize_gemini_payload,
    to_contents,
    to_plain_value,
)
from .response_parse import (
    extract_text,
    extract_tool_calls,
    setup_context_cache,
    stream_response,
    usage_from_response,
)

__all__ = [
    "to_contents",
    "sanitize_gemini_payload",
    "to_plain_value",
    "convert_tools_to_gemini",
    "extract_tool_calls",
    "extract_text",
    "usage_from_response",
    "stream_response",
    "setup_context_cache",
]
