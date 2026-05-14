"""Provider-agnostic tool definitions (facade for backward compatibility)."""

from .definitions import get_tool_definitions, normalize_mcp_tool_definitions

__all__ = ["get_tool_definitions", "normalize_mcp_tool_definitions"]
