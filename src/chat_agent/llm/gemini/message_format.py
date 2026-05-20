"""
llm/gemini/message_format.py
──────────────────────────────
Message conversion and schema sanitization utilities for the Gemini API.
  to_contents          — convert chat messages → Gemini contents format
  sanitize_gemini_payload — strip proto-incompatible JSON Schema keys
  to_plain_value       — recursively convert protobuf values → plain Python
  convert_tools_to_gemini — convert OpenAI-style tool defs → Gemini format
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keys rejected by Gemini's Protocol Buffer schema validator
_UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "additionalProperties", "patternProperties", "unevaluatedProperties",
    "propertyNames", "contains", "minProperties", "maxProperties",
    "dependencies", "pattern", "$defs", "definitions", "$schema", "$id",
})


def to_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat messages to Gemini contents format, merging consecutive same-role turns."""
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")

        if role in {"user", "system"}:
            mapped_role = "user"
        elif role == "assistant":
            mapped_role = "model"
        else:
            mapped_role = "user"   # tool results also mapped to user

        content_text = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        parts: list[dict[str, Any]] = []

        if role == "tool":
            tool_name = msg.get("name", "unknown_tool")
            content_text = (
                f"\n\n--- TOOL EXECUTION RESULT ---\n"
                f"[Tool: {tool_name}]\n{content_text}\n"
                f"-----------------------------\n"
                f"CRITICAL: You now have the data from the tool '{tool_name}'. "
                f"DO NOT call '{tool_name}' again. Use the data above to answer the user's query."
            )
            parts.append({"text": content_text})
        else:
            if content_text:
                parts.append({"text": content_text})
            for tc in tool_calls:
                tc_text = f"\n**Action**: {tc.get('name')}"
                if tc.get("arguments"):
                    try:
                        tc_text += f"\n```json\n{json.dumps(tc.get('arguments'))}\n```"
                    except Exception:
                        pass
                parts.append({"text": tc_text})

        if not parts:
            continue

        # Merge consecutive messages with the same role
        if contents and contents[-1]["role"] == mapped_role:
            for part in parts:
                if "text" in part:
                    text_part = next((p for p in contents[-1]["parts"] if "text" in p), None)
                    if text_part:
                        text_part["text"] += "\n\n" + part["text"]
                    else:
                        contents[-1]["parts"].append(part)
                else:
                    contents[-1]["parts"].append(part)
        else:
            contents.append({"role": mapped_role, "parts": parts})

    return contents


def sanitize_gemini_payload(value: Any) -> Any:
    """
    Recursively remove JSON Schema keys that Gemini's proto layer rejects.

    The Gemini SDK translates tool schemas into Protocol Buffer messages.
    Keys like 'additionalProperties' have no proto field and raise ValueError.
    """
    if isinstance(value, dict):
        return {
            k: sanitize_gemini_payload(v)
            for k, v in value.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [sanitize_gemini_payload(item) for item in value]
    return value


def to_plain_value(value: Any) -> Any:
    """Convert protobuf/map/repeated values to plain JSON-safe Python objects."""
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


def convert_tools_to_gemini(tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OpenAI-style tool definitions to Gemini function_declarations format."""
    try:
        from ...tools.schemas import ToolSchemaConverter  # type: ignore
    except (ImportError, ValueError):
        try:
            from chat_agent.tools.schemas import ToolSchemaConverter  # type: ignore
        except ImportError:
            ToolSchemaConverter = None

    function_declarations = []
    for tool in tools:
        if ToolSchemaConverter is not None:
            try:
                function_declarations.append(ToolSchemaConverter.to_gemini(tool))
                continue
            except Exception as e:
                logger.debug(f"ToolSchemaConverter failed for {tool.get('name')}: {e}")

        params = tool.get("parameters") or tool.get("inputSchema") or {}
        if ToolSchemaConverter is not None:
            try:
                params = ToolSchemaConverter._sanitize_gemini_schema(params)
            except Exception:
                pass
        function_declarations.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": params,
        })

    return sanitize_gemini_payload({"function_declarations": function_declarations})
