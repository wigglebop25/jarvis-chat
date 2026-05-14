"""Execute tools and handle errors."""

import logging
from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ...mcp.router_protocol import MCPRouterLike
from ...skills import ToolErrorRepromptSkill
from ..state import AgentState

logger = logging.getLogger(__name__)


async def execute_tools_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that executes tool calls.
    Special handling for playMusic: checks queue first to avoid clearing it.
    """
    configurable = config.get("configurable", {})
    mcp_router: Optional[MCPRouterLike] = configurable.get("mcp_router")
    
    if not mcp_router:
        raise ValueError("MCP router not found in config")
        
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}
        
    new_messages = []
    execution_error = None
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        # Special handling for playMusic to preserve queue
        if tool_name == "playMusic" and tool_args.get("type") == "track":
            from ...music_handler import play_track_smart
            track_name = tool_args.get("trackName") or tool_args.get("id", "unknown track")
            device_id = tool_args.get("deviceId")
            result = await play_track_smart(mcp_router, track_name, device_id)
        else:
            # Execute the tool normally
            result = await mcp_router.execute_tool(tool_name, **tool_args)
        
        # Format the result
        from ...tools.formatter import format_tool_result, format_tool_error
        
        if result.is_error:
            content = format_tool_error(tool_name, result.error or "Unknown error")
            execution_error = result.error or "Unknown tool error"
        else:
            content = format_tool_result(tool_name, result.result)
            
        new_messages.append(ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name
        ))
        
    return {
        "messages": new_messages,
        "execution_error": execution_error
    }


async def tool_error_reprompt_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that generates a reprompt for failed tool calls.
    """
    if not state.get("execution_error"):
        return {}
        
    configurable = config.get("configurable", {})
    reprompt_skill: Optional[ToolErrorRepromptSkill] = configurable.get("reprompt_skill")
    if not reprompt_skill:
        return {}
        
    # Find the failed tool call in history
    last_ai_message = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.tool_calls:
            last_ai_message = m
            break
            
    if not last_ai_message or not last_ai_message.tool_calls:
        return {}
        
    # For simplicity, we reprompt for the first tool call that failed
    failed_call = last_ai_message.tool_calls[0]
    
    reprompt = reprompt_skill.build_reprompt(
        failed_tool_name=failed_call["name"],
        failed_arguments=failed_call["args"],
        error_message=state.get("execution_error") or "Unknown error",
        attempt=1  # For now just 1, could track in state
    )
    
    return {
        "messages": [SystemMessage(content=reprompt)],
        "execution_error": None  # Clear the error after reprompting
    }
