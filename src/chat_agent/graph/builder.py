from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import call_model, execute_tools, handle_tool_error
from .edges import should_continue, check_for_errors

def create_graph():
    """
    Creates the LangGraph workflow with tool repair logic and persistence.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("repair", handle_tool_error)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    
    # Add conditional edges from tools
    workflow.add_conditional_edges(
        "tools",
        check_for_errors,
    )
    
    # Add edge from repair back to agent
    workflow.add_edge("repair", "agent")
    
    # Add memory for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)
