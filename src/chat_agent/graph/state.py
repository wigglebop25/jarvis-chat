from typing import Annotated, TypedDict, Any, Optional
from langgraph.graph.message import add_messages
from ..models import Intent

class AgentState(TypedDict):
    """
    The state of the LangGraph agent.
    """
    # Messages in the conversation. add_messages handles appending new messages.
    messages: Annotated[list[Any], add_messages]
    
    # The recognized intent from the user transcript.
    intent: Optional[Intent]
    
    # Metadata for the current session.
    session_id: str
    
    # Track execution errors for reprompting.
    execution_error: Optional[str]
    
    # Track the last tool call for repair/reprompt logic.
    last_tool_call_id: Optional[str]
