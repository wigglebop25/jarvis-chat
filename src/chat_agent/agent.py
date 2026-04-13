"""
Chat Agent

Main Chat Agent class that processes voice transcripts,
recognizes intents, and routes to appropriate skills via MCP.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import AgentConfig, LLMConfig
from .context_cache import SessionContextCache
from .llm import create_provider, LLMProviderError, ToolCall as ProviderToolCall
from .mcp import MCPRouter
from .models import ConversationContext, MessageRole
from .skills import ToolErrorRepromptSkill
from .tools.definitions import get_tool_definitions

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
        self.session_id = self.config.session_id
        self.context = ConversationContext()

        # Try to create LLM provider, but allow operation without it
        self.llm_provider = None
        try:
            self.llm_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
        except Exception as e:
            logger.warning(f"Could not initialize LLM provider: {e}. Using fallback mode.")

        self.mcp_router = MCPRouter()
        self.tool_error_reprompt_skill = ToolErrorRepromptSkill(
            max_retries=self.config.tool_retry_attempts,
            base_backoff_seconds=self.config.tool_retry_backoff_seconds,
        )

        self.context_cache: SessionContextCache | None = None
        if self.config.context_cache_enabled:
            persistence_path = (
                Path(self.config.context_cache_path).expanduser()
                if self.config.context_cache_path
                else None
            )
            self.context_cache = SessionContextCache(
                provider=self.llm_config.provider,
                model=self.llm_config.model,
                requested_dtype=self.config.context_dtype,
                max_turns=self.config.context_cache_max_turns,
                summary_keep_last=self.config.context_cache_summary_keep_last,
                token_budget=self.config.context_token_budget,
                persistence_path=persistence_path,
            )

        self.context.add_message(MessageRole.SYSTEM, self.config.system_prompt)

        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)

    def _record_message(self, role: MessageRole, content: str, **kwargs) -> None:
        self.context.add_message(role, content, **kwargs)
        if self.context_cache:
            self.context_cache.add_message(self.session_id, role.value, content)

    def _build_messages(self) -> list[dict[str, str]]:
        """Convert ConversationContext to message list for LLM."""
        if self.context_cache:
            return self.context_cache.build_messages(
                session_id=self.session_id,
                system_prompt=self.config.system_prompt,
            )

        messages: list[dict[str, str]] = []
        for msg in self.context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})
        return messages

    def _get_tools_payload(self) -> list[dict[str, Any]] | None:
        if self.llm_provider and self.llm_provider.supports_tools:
            return get_tool_definitions()
        return None

    def _handle_with_llm_sync(self, transcript: str) -> str:
        """
        Handle request using LLM with synchronous calls.
        
        Args:
            transcript: User's input text
            
        Returns:
            Response text from the agent
        """
        try:
            messages = self._build_messages()
            response = self.llm_provider.complete_sync(
                messages,
                tools=self._get_tools_payload(),
            )

            if response.tool_calls:
                raise RuntimeError("Provider returned tool calls in sync path.")

            if response.text:
                self._record_message(MessageRole.ASSISTANT, response.text)
                return response.text
            return ""

        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

    async def _execute_tool_with_reprompt(
        self,
        tool_call: ProviderToolCall,
    ) -> tuple[ProviderToolCall, dict[str, Any]]:
        current_call = tool_call
        attempt = 0
        tools_payload = self._get_tools_payload()

        while True:
            result = await self.mcp_router.execute_tool(
                current_call.name,
                **current_call.arguments,
            )
            result_payload = result.model_dump()
            if not result.is_error:
                return current_call, result_payload

            attempt += 1
            if attempt > self.tool_error_reprompt_skill.max_retries:
                return current_call, result_payload

            reprompt = self.tool_error_reprompt_skill.build_reprompt(
                failed_tool_name=current_call.name,
                failed_arguments=current_call.arguments,
                error_message=result.error or "unknown tool error",
                attempt=attempt,
            )
            self._record_message(MessageRole.SYSTEM, reprompt)
            self.tool_error_reprompt_skill.backoff(attempt)

            repair_response = await self.llm_provider.complete(
                self._build_messages(),
                tools=tools_payload,
            )
            if repair_response.tool_calls:
                current_call = repair_response.tool_calls[0]
                continue

            if repair_response.text:
                self._record_message(MessageRole.ASSISTANT, repair_response.text)
            return current_call, result_payload

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
            tools_payload = self._get_tools_payload()
            response = await self.llm_provider.complete(messages, tools=tools_payload)

            latest_text = ""
            if response.text:
                latest_text = response.text
                self._record_message(MessageRole.ASSISTANT, response.text)

            # Handle tool calls
            while response.tool_calls:
                for tool_call in response.tool_calls:
                    executed_call, result_payload = await self._execute_tool_with_reprompt(tool_call)
                    self._record_message(
                        MessageRole.TOOL,
                        json.dumps(result_payload, default=str),
                        tool_call_id=executed_call.id,
                        name=executed_call.name,
                    )

                # Get next response from LLM after tool execution
                messages = self._build_messages()
                response = await self.llm_provider.complete(messages, tools=tools_payload)

                if response.text:
                    latest_text = response.text
                    self._record_message(MessageRole.ASSISTANT, response.text)

            return latest_text

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

        self._record_message(MessageRole.USER, transcript)

        # Use LLM if available, otherwise use pattern matching
        if self.llm_provider:
            try:
                if self.llm_provider.supports_tools:
                    return await self._handle_with_llm(transcript)

                # Try sync first (faster, more reliable for HTTP handlers)
                try:
                    return self._handle_with_llm_sync(transcript)
                except (RuntimeError, asyncio.InvalidStateError):
                    # Fall back to async if sync fails
                    return await self._handle_with_llm(transcript)
            except (RuntimeError, LLMProviderError) as e:
                # If event loop is closed or LLM fails, use pattern matching
                error_msg = str(e)
                lowered = error_msg.lower()
                if (
                    "event loop is closed" in lowered
                    or "no running event loop" in lowered
                    or "404" in lowered
                    or "timed out" in lowered
                    or "timeout" in lowered
                ):
                    logger.warning(f"LLM unavailable ({error_msg}), using fallback intent recognition")
                    from .intent import recognize_intent
                    intent = recognize_intent(transcript)
                    
                    if intent.type.value == "unknown":
                        response = "I don't understand. Could you rephrase that?"
                    else:
                        response = f"Recognized: {intent.type.value} (confidence: {intent.confidence:.2f})"

                    self._record_message(MessageRole.ASSISTANT, response)
                    return response
                raise
        else:
            # Fallback: Use intent recognition patterns
            from .intent import recognize_intent
            intent = recognize_intent(transcript)
            
            if intent.type.value == "unknown":
                response = "I don't understand. Could you rephrase that?"
            else:
                response = f"Recognized: {intent.type.value} (confidence: {intent.confidence:.2f})"

            self._record_message(MessageRole.ASSISTANT, response)
            return response

    def clear_context(self) -> None:
        """Clear the conversation context while keeping the system prompt."""
        self.context.clear(keep_system=True)
        if self.context_cache:
            self.context_cache.clear_session(self.session_id)

    def set_session_id(self, session_id: str) -> None:
        normalized = session_id.strip() if session_id else "default"
        self.session_id = normalized or "default"

    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.context.messages.copy()

    def get_cache_stats(self) -> dict[str, Any]:
        if not self.context_cache:
            return {"enabled": False}
        stats = self.context_cache.get_stats(self.session_id)
        stats["enabled"] = True
        return stats

    def register_context_artifact(
        self,
        name: str,
        values: list[float],
        source_dtype: str = "fp32",
    ) -> dict[str, Any]:
        if not self.context_cache:
            raise RuntimeError("Context cache is disabled.")
        result = self.context_cache.register_artifact(
            session_id=self.session_id,
            name=name,
            values=values,
            source_dtype=source_dtype,
        )
        return result.to_dict()

    def convert_context_artifact_dtype(
        self,
        name: str,
        target_dtype: str,
    ) -> dict[str, Any]:
        if not self.context_cache:
            raise RuntimeError("Context cache is disabled.")
        result = self.context_cache.convert_artifact_dtype(
            session_id=self.session_id,
            name=name,
            target_dtype=target_dtype,
        )
        return result.to_dict()
