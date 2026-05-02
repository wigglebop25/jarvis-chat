"""
Chat Agent Models

Pydantic models for chat messages, intents, and tool calls.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a chat message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single chat message."""
    
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class IntentType(str, Enum):
    """Types of recognized user intents."""
    
    SYSTEM_INFO = "system_info"
    VOLUME_CONTROL = "volume_control"
    MUSIC_CONTROL = "music_control"
    NETWORK_TOGGLE = "network_toggle"
    DIRECTORY_LIST = "directory_list"
    FILE_ORGANIZATION = "file_organization"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"


class Intent(BaseModel):
    """Recognized intent from user input."""
    
    type: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""


class ToolCall(BaseModel):
    """A tool call to be executed by the MCP Server."""
    
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from executing a tool."""
    
    tool_call_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """Complete response from the Chat Agent."""
    
    text: str
    intent: Optional[Intent] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class ConversationContext(BaseModel):
    """Context for an ongoing conversation."""
    
    messages: list[ChatMessage] = Field(default_factory=list)
    max_history: int = Field(default=20)
    
    def add_message(self, role: MessageRole, content: str, **kwargs) -> None:
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role=role, content=content, **kwargs))
        
        if len(self.messages) > self.max_history:
            system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_msgs = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            other_msgs = other_msgs[-(self.max_history - len(system_msgs)):]
            self.messages = system_msgs + other_msgs
    
    def get_openai_messages(self) -> list[dict]:
        """Convert messages to OpenAI format."""
        return [
            {"role": m.role.value, "content": m.content}
            for m in self.messages
        ]
    
    def clear(self, keep_system: bool = True) -> None:
        """Clear conversation history."""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        else:
            self.messages = []
