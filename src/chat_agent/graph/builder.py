from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    route_transcript_node, 
    generate_response_node, 
    execute_tools_node, 
    tool_error_reprompt_node, 
    classify_intent_node
)
from .edges import route_after_router, should_continue, check_for_errors

def create_graph():
    """
    Creates the LangGraph workflow following the standard ReAct framework.
    
    The cycle is:
    [Router/Intent] -> [RAG Context] -> [Agent Reasoning/Generation] -> [Tools Execution] -> [Response]
                                           ^                                  |
                                           |----------------------------------| (Loop on tool calls)
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes - named according to their functional role in the ReAct framework
    workflow.add_node("intent_router", route_transcript_node)
    workflow.add_node("rag_retrieval", classify_intent_node)
    workflow.add_node("llm_reasoning", generate_response_node)
    workflow.add_node("tool_execution", execute_tools_node)
    workflow.add_node("tool_repair", tool_error_reprompt_node)
    
    # Set entry point
    workflow.set_entry_point("intent_router")
    
    # 1. Router -> RAG (if not fast-path) or END (if fast-path)
    workflow.add_conditional_edges(
        "intent_router",
        route_after_router,
        {
            "agent": "rag_retrieval",
            "__end__": "__end__"
        }
    )
    
    # 2. RAG -> LLM Reasoning
    workflow.add_edge("rag_retrieval", "llm_reasoning")
    
    # 3. LLM Reasoning -> Tool Execution or END
    workflow.add_conditional_edges(
        "llm_reasoning",
        should_continue,
        {
            "tools": "tool_execution",
            "__end__": "__end__"
        }
    )
    
    # 4. Tool Execution -> Tool Repair or back to LLM Reasoning (Observation)
    workflow.add_conditional_edges(
        "tool_execution",
        check_for_errors,
        {
            "repair": "tool_repair",
            "agent": "llm_reasoning"
        }
    )
    
    # 5. Tool Repair -> back to LLM Reasoning
    workflow.add_edge("tool_repair", "llm_reasoning")
    
    # Add memory for persistence
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)
