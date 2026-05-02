"""
Chat Agent

Main Chat Agent class that processes voice transcripts,
recognizes intents, and routes to appropriate skills via MCP.
"""

import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any, Optional

from .config import AgentConfig, LLMConfig
from .context_cache import SessionContextCache
from .llm import create_provider, LLMProviderError, ToolCall as ProviderToolCall
from .mcp import MCPRouter
from .models import ConversationContext, Intent, IntentType, MessageRole
from .response_cache import LLMResponseCache
from .skills import ToolErrorRepromptSkill
from .tools.definitions import get_tool_definitions
from .tools.formatter import format_tool_result, format_tool_error
from .message_builder import MessageBuilder
from .tool_discovery import ToolDiscovery

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

        # Initialize MCP router with configured host/port
        from .mcp.client import MCPClient
        mcp_client = MCPClient(base_url=self.config.mcp.url)
        self.mcp_router = MCPRouter(mcp_client=mcp_client)
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

        self.response_cache: LLMResponseCache | None = None
        if self.config.llm_response_cache_enabled:
            response_cache_path = (
                Path(self.config.llm_response_cache_path).expanduser()
                if self.config.llm_response_cache_path
                else None
            )
            self.response_cache = LLMResponseCache(
                ttl_seconds=self.config.llm_response_cache_ttl_seconds,
                max_entries=self.config.llm_response_cache_max_entries,
                min_chars=self.config.llm_response_cache_min_chars,
                persistence_path=response_cache_path,
            )

        self.context.add_message(MessageRole.SYSTEM, self.config.system_prompt)
        self._tool_definitions = get_tool_definitions()
        
        # Initialize message builder and tool discovery
        self.message_builder = MessageBuilder(self.context, self._tool_definitions)
        self.tool_discovery = ToolDiscovery(self.mcp_router, self._tool_definitions)

        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)

    def _record_message(self, role: MessageRole, content: str, **kwargs) -> None:
        """Record a message in the conversation context."""
        self.message_builder.record_message(role, content, **kwargs)
        if self.context_cache:
            self.context_cache.add_message(self.session_id, role.value, content)

    def _build_messages(self) -> list[dict[str, str]]:
        """Convert ConversationContext to message list for LLM."""
        if self.context_cache:
            return self.context_cache.build_messages(
                session_id=self.session_id,
                system_prompt=self.config.system_prompt,
            )

        return self.message_builder.build_messages()

    def _get_tools_payload(self) -> list[dict[str, Any]] | None:
        """Get tool definitions if the provider supports tools."""
        if self.llm_provider and self.llm_provider.supports_tools:
            return self.message_builder.get_tools_payload()
        return None

    def _response_cache_lookup(
        self,
        *,
        transcript: str,
        intent: Intent | None,
        messages: list[dict[str, str]],
        tools_payload: list[dict[str, Any]] | None,
    ) -> tuple[str | None, tuple[str, str, str] | None]:
        if not self.response_cache or not self.llm_provider:
            return None, None

        decision = self.response_cache.evaluate_eligibility(
            transcript=transcript,
            intent_type=intent.type.value if intent else None,
            supports_tools=self.llm_provider.supports_tools,
            tools_payload=tools_payload,
            allow_tool_providers=self.config.llm_response_cache_allow_tool_providers,
            messages=messages,
        )
        if not decision.cacheable:
            self.response_cache.record_skip(decision.reason)
            return None, None

        key_bundle = self.response_cache.build_key(
            transcript=transcript,
            session_id=self.session_id,
            provider=self.llm_config.provider,
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            system_prompt=self.config.system_prompt,
            tools_payload=tools_payload,
        )
        cached_text = self.response_cache.lookup(key_bundle[0])
        return cached_text, key_bundle

    def _response_cache_store(
        self,
        *,
        key_bundle: tuple[str, str, str] | None,
        response_text: str,
        latency_ms: float,
        used_tools: bool,
    ) -> None:
        if not self.response_cache or not key_bundle:
            return
        if used_tools:
            self.response_cache.record_skip("tool_invocation")
            return

        store_decision = self.response_cache.should_store_response(response_text)
        if not store_decision.cacheable:
            self.response_cache.record_skip(store_decision.reason)
            return

        cache_key, transcript_key, tools_fingerprint = key_bundle
        self.response_cache.store(
            key=cache_key,
            response_text=response_text,
            source_latency_ms=latency_ms,
            session_id=self.session_id,
            provider=self.llm_config.provider,
            model=self.llm_config.model,
            transcript_key=transcript_key,
            tools_fingerprint=tools_fingerprint,
        )

    async def _refresh_tool_definitions(self) -> None:
        """Refresh tool definitions from MCP server asynchronously."""
        supports_tools = bool(self.llm_provider and self.llm_provider.supports_tools)
        tool_defs_changed = await self.tool_discovery.refresh_async(supports_tools)
        
        if tool_defs_changed:
            self._tool_definitions = self.tool_discovery.tools
            self.message_builder.tool_definitions = self._tool_definitions
            if self.response_cache:
                self.response_cache.invalidate_all()

    def _refresh_tool_definitions_sync(self) -> None:
        """Refresh tool definitions from MCP server synchronously."""
        supports_tools = bool(self.llm_provider and self.llm_provider.supports_tools)
        tool_defs_changed = self.tool_discovery.refresh_sync(supports_tools)
        
        if tool_defs_changed:
            self._tool_definitions = self.tool_discovery.tools
            self.message_builder.tool_definitions = self._tool_definitions
            if self.response_cache:
                self.response_cache.invalidate_all()

    def _handle_with_llm_sync(self, transcript: str, intent: Intent | None = None) -> str:
        """
        Handle request using LLM with synchronous calls.
        
        Args:
            transcript: User's input text
            
        Returns:
            Response text from the agent
        """
        try:
            self._refresh_tool_definitions_sync()
            messages = self._build_messages()
            tools_payload = self._get_tools_payload()
            cached_text, key_bundle = self._response_cache_lookup(
                transcript=transcript,
                intent=intent,
                messages=messages,
                tools_payload=tools_payload,
            )
            if cached_text is not None:
                self._record_message(MessageRole.ASSISTANT, cached_text)
                return cached_text

            if not self.llm_provider:
                raise RuntimeError("LLM provider not initialized for sync completion")
            request_started = time.perf_counter()
            response = self.llm_provider.complete_sync(messages, tools=tools_payload)
            latency_ms = (time.perf_counter() - request_started) * 1000.0

            if response.tool_calls:
                raise RuntimeError("Provider returned tool calls in sync path.")

            if response.text:
                self._record_message(MessageRole.ASSISTANT, response.text)
                self._response_cache_store(
                    key_bundle=key_bundle,
                    response_text=response.text,
                    latency_ms=latency_ms,
                    used_tools=False,
                )
                return response.text
            self._response_cache_store(
                key_bundle=key_bundle,
                response_text=response.text,
                latency_ms=latency_ms,
                used_tools=False,
            )
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
            # Ensure 'name' parameter isn't duplicated in arguments
            args = {k: v for k, v in current_call.arguments.items() if k != 'name'}
            result = await self.mcp_router.execute_tool(
                current_call.name,
                **args,
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

            if not self.llm_provider:
                raise RuntimeError("LLM provider not initialized for tool repair")
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

    async def _handle_with_llm(self, transcript: str, intent: Intent | None = None) -> str:
        """
        Handle request using LLM with tool support.
        
        Args:
            transcript: User's input text
            
        Returns:
            Response text from the agent
        """
        try:
            await self._refresh_tool_definitions()
            messages = self._build_messages()
            tools_payload = self._get_tools_payload()
            cached_text, key_bundle = self._response_cache_lookup(
                transcript=transcript,
                intent=intent,
                messages=messages,
                tools_payload=tools_payload,
            )
            if cached_text is not None:
                self._record_message(MessageRole.ASSISTANT, cached_text)
                return cached_text

            request_started = time.perf_counter()
            if not self.llm_provider:
                raise RuntimeError("LLM provider not initialized for async completion")
            response = await self.llm_provider.complete(messages, tools=tools_payload)
            latency_ms = (time.perf_counter() - request_started) * 1000.0

            latest_text = ""
            used_tools = bool(response.tool_calls)
            
            # If there are tool calls, skip the reasoning text and only keep the final response
            if response.tool_calls:
                latest_text = ""  # Reset - we'll use the final response after tool execution
            elif response.text:
                latest_text = response.text
                self._record_message(MessageRole.ASSISTANT, response.text)

            # Handle tool calls
            while response.tool_calls:
                used_tools = True
                for tool_call in response.tool_calls:
                    executed_call, result_payload = await self._execute_tool_with_reprompt(tool_call)
                    
                    # Format the tool result for the LLM
                    formatted_result = format_tool_result(executed_call.name, result_payload.get("result") if isinstance(result_payload, dict) else result_payload)
                    
                    self._record_message(
                        MessageRole.TOOL,
                        formatted_result,
                        tool_call_id=executed_call.id,
                        name=executed_call.name,
                    )

                # Get next response from LLM after tool execution
                messages = self._build_messages()
                if not self.llm_provider:
                    raise RuntimeError("LLM provider lost during tool reprompt")
                response = await self.llm_provider.complete(messages, tools=tools_payload)

                if response.text:
                    latest_text = response.text
                    self._record_message(MessageRole.ASSISTANT, response.text)

            self._response_cache_store(
                key_bundle=key_bundle,
                response_text=latest_text,
                latency_ms=latency_ms,
                used_tools=used_tools,
            )
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

        # Get route decision and optionally execute from Rust MCP server
        route_result = await self.mcp_router.route_and_call(transcript)
        
        intent_str = route_result.get("intent", "UNKNOWN").lower()
        confidence = route_result.get("confidence", 0.0)
        try:
            intent_type = IntentType(intent_str)
        except ValueError:
            intent_type = IntentType.UNKNOWN
            
        intent = Intent(
            type=intent_type,
            confidence=confidence,
            parameters=route_result.get("arguments", {}),
            raw_text=transcript
        )

        if route_result.get("should_execute") and route_result.get("tool_name"):
            tool_name = str(route_result.get("tool_name", "unknown"))
            if "execution_error" in route_result:
                error_msg = str(route_result["execution_error"])
                payload = {"is_error": True, "error": error_msg}
                self._record_message(MessageRole.TOOL, json.dumps(payload, default=str), tool_call_id=f"intent-{len(self.context.messages)}", name=tool_name)
                response = format_tool_error(tool_name, error_msg)
            else:
                result_data = route_result.get("execution_result")
                payload = {"is_error": False, "result": result_data}
                self._record_message(MessageRole.TOOL, json.dumps(payload, default=str), tool_call_id=f"intent-{len(self.context.messages)}", name=tool_name)
                response = format_tool_result(tool_name, result_data)
                
            self._record_message(MessageRole.ASSISTANT, response)
            return response

        # Use LLM if available, otherwise use pattern matching
        if self.llm_provider:
            try:
                if self.llm_provider.supports_tools:
                    return await self._handle_with_llm(transcript, intent=intent)

                # Try sync first (faster, more reliable for HTTP handlers)
                try:
                    return self._handle_with_llm_sync(transcript, intent=intent)
                except (RuntimeError, asyncio.InvalidStateError):
                    # Fall back to async if sync fails
                    return await self._handle_with_llm(transcript, intent=intent)
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
                    if intent.type.value == "unknown":
                        response = "I don't understand. Could you rephrase that?"
                    else:
                        response = f"Recognized: {intent.type.value} (confidence: {intent.confidence:.2f})"

                    self._record_message(MessageRole.ASSISTANT, response)
                    return response
                raise
        else:
            # Fallback: Use intent recognition patterns
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
        if self.response_cache:
            self.response_cache.invalidate_session(self.session_id)

    def set_session_id(self, session_id: str) -> None:
        normalized = session_id.strip() if session_id else "default"
        self.session_id = normalized or "default"

    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.context.messages.copy()

    def get_cache_stats(self) -> dict[str, Any]:
        if self.context_cache:
            stats: dict[str, Any] = self.context_cache.get_stats(self.session_id)
            stats["enabled"] = True
        else:
            stats = {"enabled": False}

        if self.response_cache:
            stats["response_cache"] = self.response_cache.get_stats()
        else:
            stats["response_cache"] = {"enabled": False}
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
