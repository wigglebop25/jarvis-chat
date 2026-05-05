"""
Chat Agent

Main Chat Agent class that processes voice transcripts,
recognizes intents, and routes to appropriate skills via MCP.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import AgentConfig, LLMConfig
from .context_cache import SessionContextCache
from .llm import create_provider
from .mcp import MCPRouter
from .models import ConversationContext, Intent, IntentType, MessageRole
from .response_cache import LLMResponseCache
from .skills import ToolErrorRepromptSkill
from .tools.definitions import get_tool_definitions
from .tools.formatter import format_tool_result, format_tool_error
from .message_builder import MessageBuilder
from .tool_discovery import ToolDiscovery
from .agent_logic.mixins import LLMHandlerMixin, ToolHandlerMixin, CacheHandlerMixin

logger = logging.getLogger(__name__)


class ChatAgent(LLMHandlerMixin, ToolHandlerMixin, CacheHandlerMixin):
    """
    JARVIS Chat Agent

    Processes voice transcripts using modular LLM providers
    and routes tool calls through MCP Server.
    Can operate in legacy mode (imperative) or langgraph mode (graph-based).
    """

    def __init__(self, config: Optional[AgentConfig] = None, llm_config: Optional[LLMConfig] = None):
        self.config = config or AgentConfig()
        self.llm_config = llm_config or self.config.llm
        self.session_id = self.config.session_id
        
        if self.config.agent_type == "langgraph":
            from .langgraph_agent import LangGraphChatAgent
            self._delegate = LangGraphChatAgent(self.config, llm_config)
            return

        self.context = ConversationContext()

        self.llm_provider = None
        try:
            self.llm_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
        except Exception as e:
            logger.warning(f"Could not initialize LLM provider: {e}. Using fallback mode.")

        from .mcp.client import MCPClient
        mcp_client = MCPClient(base_url=self.config.mcp.url)
        self.mcp_router = MCPRouter(mcp_client=mcp_client)
        self.tool_error_reprompt_skill = ToolErrorRepromptSkill(
            max_retries=self.config.tool_retry_attempts,
            base_backoff_seconds=self.config.tool_retry_backoff_seconds,
        )

        self.context_cache = None
        if self.config.context_cache_enabled and self.llm_provider:
            path = Path(self.config.context_cache_path).expanduser() if self.config.context_cache_path else None
            self.context_cache = SessionContextCache(
                llm_provider=self.llm_provider,
                requested_dtype=self.config.context_dtype, max_turns=self.config.context_cache_max_turns,
                summary_keep_last=self.config.context_cache_summary_keep_last,
                token_budget=self.config.context_token_budget, persistence_path=path,
            )

        self.response_cache = None
        if self.config.llm_response_cache_enabled:
            path = Path(self.config.llm_response_cache_path).expanduser() if self.config.llm_response_cache_path else None
            self.response_cache = LLMResponseCache(
                ttl_seconds=self.config.llm_response_cache_ttl_seconds,
                max_entries=self.config.llm_response_cache_max_entries,
                min_chars=self.config.llm_response_cache_min_chars, persistence_path=path,
            )

        self.context.add_message(MessageRole.SYSTEM, self.config.system_prompt)
        self._tool_definitions = get_tool_definitions()
        self.message_builder = MessageBuilder(self.context, self._tool_definitions)
        self.tool_discovery = ToolDiscovery(self.mcp_router, self._tool_definitions)

    def _record_message(self, role: MessageRole, content: str, **kwargs) -> None:
        self.message_builder.record_message(role, content, **kwargs)
        if self.context_cache:
            self.context_cache.add_message(self.session_id, role.value, content)

    def _build_messages(self) -> list[dict[str, str]]:
        if self.context_cache:
            return self.context_cache.build_messages(self.session_id, self.config.system_prompt)
        return self.message_builder.build_messages()

    def _get_tools_payload(self) -> list[dict[str, Any]] | None:
        if self.llm_provider and self.llm_provider.supports_tools:
            return self.message_builder.get_tools_payload()
        return None

    async def process_transcript(self, transcript: str) -> str:
        if hasattr(self, "_delegate"):
            return await self._delegate.process_transcript(transcript)

        if not transcript or not transcript.strip():
            return "I didn't catch that. Could you please repeat?"

        if self.config.log_transcripts:
            logger.info(f"Processing transcript: {transcript}")

        self._record_message(MessageRole.USER, transcript)
        route_result = await self.mcp_router.route_and_call(transcript)

        intent_str = route_result.get("intent", "UNKNOWN").lower()
        confidence = route_result.get("confidence", 0.0)
        try:
            intent_type = IntentType(intent_str)
        except ValueError:
            intent_type = IntentType.UNKNOWN

        intent = Intent(type=intent_type, confidence=confidence,
                       parameters=route_result.get("arguments", {}), raw_text=transcript)

        if route_result.get("should_execute") and route_result.get("tool_name"):
            tool_name = str(route_result.get("tool_name", "unknown"))
            if "execution_error" in route_result:
                error_msg = str(route_result["execution_error"])
                self._record_message(MessageRole.TOOL, json.dumps({"is_error": True, "error": error_msg}),
                                   tool_call_id=f"intent-{len(self.context.messages)}", name=tool_name)
                response = format_tool_error(tool_name, error_msg)
            else:
                result_data = route_result.get("execution_result")
                self._record_message(MessageRole.TOOL, json.dumps({"is_error": False, "result": result_data}),
                                   tool_call_id=f"intent-{len(self.context.messages)}", name=tool_name)
                response = format_tool_result(tool_name, result_data)

            self._record_message(MessageRole.ASSISTANT, response)
            return response

        if self.llm_provider:
            try:
                if self.llm_provider.supports_tools:
                    return await self._handle_with_llm(transcript, intent=intent)
                try:
                    return self._handle_with_llm_sync(transcript, intent=intent)
                except Exception:
                    return await self._handle_with_llm(transcript, intent=intent)
            except Exception as e:
                logger.warning(f"LLM failure: {e}, using fallback")
                response = "I don't understand." if intent.type == IntentType.UNKNOWN else f"Recognized: {intent.type.value}"
                self._record_message(MessageRole.ASSISTANT, response)
                return response

        response = "I don't understand." if intent.type == IntentType.UNKNOWN else f"Recognized: {intent.type.value}"
        self._record_message(MessageRole.ASSISTANT, response)
        return response

    def clear_context(self) -> None:
        if hasattr(self, "_delegate"):
            return self._delegate.clear_context()
        self.context.clear(keep_system=True)
        if self.context_cache: self.context_cache.clear_session(self.session_id)
        if self.response_cache: self.response_cache.invalidate_session(self.session_id)

    def change_model(self, model_name: str, provider_name: Optional[str] = None) -> None:
        """Dynamically change the LLM model and optionally the provider."""
        if hasattr(self, "_delegate"):
            return self._delegate.change_model(model_name, provider_name)
        if provider_name:
            self.llm_config.provider = provider_name
        self.llm_config.model = model_name

        new_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
        
        if not new_provider.is_configured():
            raise ValueError(f"Provider {self.llm_config.provider} is not properly configured (check API keys).")
            
        self.llm_provider = new_provider
        
        if self.context_cache:
            self.context_cache.update_provider(self.llm_provider)

    def set_session_id(self, session_id: str) -> None:
        if hasattr(self, "_delegate"):
            return self._delegate.set_session_id(session_id)
        self.session_id = session_id.strip() if session_id else "default"

    def get_conversation_history(self) -> list:
        if hasattr(self, "_delegate"):
            return self._delegate.get_conversation_history()
        return self.context.messages.copy()

    def get_cache_stats(self) -> dict[str, Any]:
        if hasattr(self, "_delegate"):
            return self._delegate.get_cache_stats()
        stats: dict[str, Any] = self.context_cache.get_stats(self.session_id) if self.context_cache else {"enabled": False}
        stats["response_cache"] = self.response_cache.get_stats() if self.response_cache else {"enabled": False}
        return stats

    def register_context_artifact(self, name: str, values: list[float], source_dtype: str = "fp32") -> dict:
        if hasattr(self, "_delegate"):
            return self._delegate.register_context_artifact(name, values, source_dtype)
        if not self.context_cache: raise RuntimeError("Context cache disabled.")
        return self.context_cache.register_artifact(self.session_id, name, values, source_dtype).to_dict()

    def convert_context_artifact_dtype(self, name: str, target_dtype: str) -> dict:
        if hasattr(self, "_delegate"):
            return self._delegate.convert_context_artifact_dtype(name, target_dtype)
        if not self.context_cache: raise RuntimeError("Context cache disabled.")
        return self.context_cache.convert_artifact_dtype(self.session_id, name, target_dtype).to_dict()
