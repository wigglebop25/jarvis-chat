"""
Graph nodes for LangGraph agent.
This module re-exports nodes from the nodes subpackage for backward compatibility.
"""

from .nodes import (
    route_transcript_node,
    classify_intent_node,
    generate_response_node,
    execute_tools_node,
    tool_error_reprompt_node,
)

# Keep legacy names for backward compatibility
router_node = route_transcript_node
retrieve_context_node = classify_intent_node
call_model = generate_response_node
execute_tools = execute_tools_node
handle_tool_error = tool_error_reprompt_node

__all__ = [
    # New names
    "route_transcript_node",
    "classify_intent_node",
    "generate_response_node",
    "execute_tools_node",
    "tool_error_reprompt_node",
    # Legacy names
    "router_node",
    "retrieve_context_node",
    "call_model",
    "execute_tools",
    "handle_tool_error",
]
