"""
Message building and management utilities.

Handles construction and recording of conversation messages
for the chat pipeline.
"""

from typing import Any, Optional

from .models import ConversationContext, MessageRole


class MessageBuilder:
    """Manages message building and recording for the chat context."""
    
    def __init__(self, context: ConversationContext, tool_definitions: list[dict[str, Any]] | None = None):
        """
        Initialize message builder.
        
        Args:
            context: ConversationContext instance
            tool_definitions: Optional tool definitions for payload
        """
        self.context = context
        self.tool_definitions = tool_definitions or []
    
    def build_messages(self) -> list[dict[str, str]]:
        """
        Build message list for LLM from conversation context.
        
        Returns:
            List of messages in [{'role': '...', 'content': '...'}] format
        """
        messages: list[dict[str, str]] = []
        
        # Add all context messages
        for msg in self.context.messages:
            entry = {
                "role": msg.role.value,
                "content": msg.content,
            }
            
            # Add tool context if applicable
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            
            messages.append(entry)
        
        return messages
    
    def record_message(
        self,
        role: MessageRole,
        content: str,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Record a message in the conversation context.
        
        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            tool_call_id: Optional tool call ID (for tool messages)
            name: Optional tool name (for tool messages)
        """
        self.context.add_message(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
    
    def get_tools_payload(self) -> list[dict[str, Any]] | None:
        """
        Build tool definitions payload for LLM providers.
        
        Returns:
            List of tool definitions or None if no tools
        """
        if not self.tool_definitions:
            return None
        
        return self.tool_definitions


# Convenience functions for direct use without class

def build_messages_from_context(context: ConversationContext) -> list[dict[str, str]]:
    """
    Build message list directly from context.
    
    Args:
        context: ConversationContext instance
    
    Returns:
        List of messages
    """
    builder = MessageBuilder(context)
    return builder.build_messages()


def record_message_to_context(
    context: ConversationContext,
    role: MessageRole,
    content: str,
    tool_call_id: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """
    Record a message directly to context.
    
    Args:
        context: ConversationContext instance
        role: Message role
        content: Message content
        tool_call_id: Optional tool call ID
        name: Optional tool name
    """
    builder = MessageBuilder(context)
    builder.record_message(role, content, tool_call_id, name)
