from typing import Literal
from langchain_core.messages import AIMessage
from .state import AgentState

import logging

logger = logging.getLogger(__name__)

def route_after_router(state: AgentState) -> Literal["agent", "__end__"]:
    """
    Determines if the graph should go to the LLM agent or end (if fast path was executed).
    """
    if state.get("fast_path_executed"):
        logger.info("[LangGraph] Router fast-path executed. Ending graph.")
        return "__end__"
    logger.info("[LangGraph] Routing to agent reasoning node.")
    return "agent"

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Determines if the graph should continue to tool execution or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"[LangGraph] Last message has {len(last_message.tool_calls)} tool calls. Routing to tools.")
        return "tools"
    
    logger.info("[LangGraph] Last message has no tool calls. Ending graph.")
    return "__end__"

def check_for_errors(state: AgentState) -> Literal["repair", "agent"]:
    """
    Checks if there were execution errors that need repair.
    """
    if state.get("execution_error"):
        return "repair"
    return "agent"
