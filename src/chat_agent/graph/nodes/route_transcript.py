"""Route transcript and detect intent."""

import logging
from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ...mcp.router_protocol import MCPRouterLike
from ..state import AgentState
from .helpers import get_transcript_from_state

logger = logging.getLogger(__name__)


async def route_transcript_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that runs the deterministic Rust hot-path router.
    Routes transcript and attempts fast-path execution.
    """
    logger.info("[LangGraph] Entering route_transcript_node")
    configurable = config.get("configurable", {})
    mcp_router: Optional[MCPRouterLike] = configurable.get("mcp_router")
    
    if not mcp_router:
        logger.warning("[LangGraph] MCP router not found in config")
        return {}
        
    transcript = get_transcript_from_state(state)
    logger.info(f"[LangGraph] Routing transcript: {transcript}")
    
    route_result = await mcp_router.route_and_call(transcript)
    logger.info(f"[LangGraph] Route result: {route_result}")
    
    intent_str = route_result.get("intent", "UNKNOWN").lower()
    confidence = route_result.get("confidence", 0.0)
    from ...models import IntentType
    try:
        intent_type = IntentType(intent_str)
    except ValueError:
        intent_type = IntentType.UNKNOWN

    intent = {
        "type": intent_type.value,
        "confidence": float(confidence),
        "parameters": route_result.get("arguments", {}),
        "raw_text": transcript,
    }
    
    if route_result.get("should_execute") and route_result.get("tool_name"):
        tool_name = str(route_result.get("tool_name", "unknown"))
        logger.info(f"[LangGraph] Fast path execution: {tool_name}")
        tool_call_id = f"intent-{len(state['messages'])}"
        
        from ...tools.formatter import format_tool_result, format_tool_error
        
        if "execution_error" in route_result:
            error_msg = str(route_result["execution_error"])
            content = format_tool_error(tool_name, error_msg)
        else:
            result_data = route_result.get("execution_result")
            content = format_tool_result(tool_name, result_data)
            
        # If content is empty for some reason, ensure we show something
        if not content:
            content = f"Successfully executed {tool_name}."

        fake_ai_msg = AIMessage(content="", tool_calls=[{
            "name": tool_name,
            "args": route_result.get("arguments", {}),
            "id": tool_call_id
        }])
        tool_msg = ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
        final_ai_msg = AIMessage(content=content)
        
        return {
            "messages": [fake_ai_msg, tool_msg, final_ai_msg],
            "intent": intent,
            "fast_path_executed": True,
            "usage": {"cache_hit": False, "prompt_tokens": 0, "completion_tokens": 0}
        }
        
    return {"intent": intent, "fast_path_executed": False}
