from typing import Annotated, TypedDict, Any, Optional, Dict
from langgraph.graph.message import add_messages


class IntentState(TypedDict):
    """Primitive-only intent snapshot persisted in checkpoints."""
    type: str
    confidence: float
    parameters: Dict[str, Any]
    raw_text: str

class AgentState(TypedDict):
    """
    The state of the LangGraph agent.
    """
    # Messages in the conversation. add_messages handles appending new messages.
    messages: Annotated[list[Any], add_messages]
    
    # The recognized intent from the user transcript.
    intent: Optional[IntentState]
    
    # Metadata for the current session.
    session_id: str
    
    # Track execution errors for reprompting.
    execution_error: Optional[str]
    
    # Track the last tool call for repair/reprompt logic.
    last_tool_call_id: Optional[str]
    
    # Track if the deterministic fast path was executed
    fast_path_executed: Optional[bool]
    
    # Track LLM token usage stats
    usage: Optional[Dict[str, Any]]
    
    # RAG context for injection into prompt
    rag_context: Optional[str]

