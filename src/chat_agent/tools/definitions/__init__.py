"""Tool definitions registry and normalization."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .system import SYSTEM_TOOLS
from .spotify import SPOTIFY_TOOLS
from .filesystem import FILE_TOOLS


_DEFAULT_TOOL_DEFINITIONS = SYSTEM_TOOLS + SPOTIFY_TOOLS + FILE_TOOLS


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get list of available tool definitions."""
    return deepcopy(_DEFAULT_TOOL_DEFINITIONS)


def normalize_mcp_tool_definitions(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize MCP tools/list payloads into provider-agnostic definitions."""
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue

        name: Any = None
        description: Any = ""
        parameters: Any = {}

        function_payload = raw_tool.get("function")
        if isinstance(function_payload, dict):
            name = function_payload.get("name")
            description = function_payload.get("description", "")
            parameters = function_payload.get("parameters", {})
        else:
            name = raw_tool.get("name")
            description = raw_tool.get("description", "")
            parameters = raw_tool.get("parameters")
            if parameters is None:
                parameters = raw_tool.get("inputSchema", {})

        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        if not clean_name or clean_name in seen_names:
            continue

        if not isinstance(description, str):
            description = str(description)
        if not isinstance(parameters, dict):
            parameters = {}

        normalized.append({"name": clean_name, "description": description, "parameters": parameters})
        seen_names.add(clean_name)

    return normalized
