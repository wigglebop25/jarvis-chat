"""
Chat Agent

Main Chat Agent class that processes voice transcripts,
recognizes intents, and routes to appropriate skills via MCP.
"""

import json
import logging
from typing import Optional

from .config import AgentConfig, LLMConfig
from .llm import create_provider, LLMProvider, LLMResponse
from .mcp import MCPRouter
from .models import ToolCall, ConversationContext, MessageRole

logger = logging.getLogger(__name__)


class ChatAgent:
    """
    JARVIS Chat Agent
    
    Processes voice transcripts using modular LLM providers
    and routes tool calls through MCP Server.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, llm_config: Optional[LLMConfig] = None):
        """
        Initialize the Chat Agent.
        
        Args:
            config: Optional AgentConfig instance
            llm_config: Optional LLMConfig for provider selection
        """
        self.config = config or AgentConfig()
        self.llm_config = llm_config or self.config.llm
        self.context = ConversationContext()
        
        self.llm_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
        self.mcp_router = MCPRouter()
        
        self.context.add_message(MessageRole.SYSTEM, self.config.system_prompt)
        
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)
    
    def _build_messages(self) -> list[dict[str, str]]:
        """Convert ConversationContext to message list for LLM."""
        messages = []
        for msg in self.context.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return messages
    
    async def _handle_with_llm(self, transcript: str) -> str:
        """
        Handle request using LLM with tool support.
        
        Args:
            transcript: User's input text
            
        Returns:
            Response text from the agent
        """
        try:
            messages = self._build_messages()
            response = await self.llm_provider.complete(messages)
            
            if response.text:
                self.context.add_message(MessageRole.ASSISTANT, response.text)
            
            # Handle tool calls
            while response.tool_calls:
                for tool_call in response.tool_calls:
                    # Execute tool via MCP router
                    result = await self.mcp_router.execute_tool(tool_call.name, tool_call.arguments)
                    self.context.add_message(
                        MessageRole.TOOL,
                        json.dumps(result),
                        tool_call_id=tool_call.id,
                        name=tool_call.name
                    )
                
                # Get next response from LLM after tool execution
                messages = self._build_messages()
                response = await self.llm_provider.complete(messages)
                
                if response.text:
                    self.context.add_message(MessageRole.ASSISTANT, response.text)
            
            return self.context.messages[-1].content if self.context.messages else ""
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise
    
    async def process_transcript(self, transcript: str) -> str:
        """
        Process a voice transcript and generate a response.
        
        Args:
            transcript: The transcribed text from voice input
            
        Returns:
            Response text
        """
        if not transcript or not transcript.strip():
            return "I didn't catch that. Could you please repeat?"
        
        if self.config.log_transcripts:
            logger.info(f"Processing transcript: {transcript}")
        
        self.context.add_message(MessageRole.USER, transcript)
        
        return await self._handle_with_llm(transcript)
    
    def clear_context(self) -> None:
        """Clear the conversation context while keeping the system prompt."""
        self.context.clear(keep_system=True)
    
    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.context.messages.copy()
