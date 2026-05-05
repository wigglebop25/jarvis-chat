from typing import Literal
from langchain_core.messages import AIMessage
from .state import AgentState

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Determines if the graph should continue to tool execution or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    return "__end__"

def check_for_errors(state: AgentState) -> Literal["repair", "agent"]:
    """
    Checks if there were execution errors that need repair.
    """
    if state.get("execution_error"):
        return "repair"
    return "agent"
