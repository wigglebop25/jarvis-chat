"""
Agent Logic Mixins

Separated logic for LLM handling, tool execution, and caching.
Uses class-level annotations to satisfy static analysis (Pylance)
without violating self-type variance rules.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from ..models import Intent, MessageRole

if TYPE_CHECKING:
    from ..llm import ToolCall as ProviderToolCall, LLMProvider
    from ..config import AgentConfig, LLMConfig
    from ..response_cache import LLMResponseCache
    from ..message_builder import MessageBuilder
    from ..tool_discovery import ToolDiscovery
    from ..skills import ToolErrorRepromptSkill
    from ..mcp import MCPRouter

logger = logging.getLogger(__name__)


class LLMHandlerMixin:
    """Mixin for LLM interaction logic."""
    # Type hints for attributes provided by the host class
    llm_provider: Optional[LLMProvider]
    llm_config: LLMConfig
    config: AgentConfig
    session_id: str
    response_cache: Optional[LLMResponseCache]

    def _handle_with_llm_sync(self: Any, transcript: str, intent: Intent | None = None) -> str:
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
                response_text="",
                latency_ms=latency_ms,
                used_tools=False,
            )
            return ""

        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

    async def _handle_with_llm(self: Any, transcript: str, intent: Intent | None = None) -> str:
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

            if response.tool_calls:
                latest_text = ""
            elif response.text:
                latest_text = response.text
                self._record_message(MessageRole.ASSISTANT, response.text)

            while response.tool_calls:
                used_tools = True
                for tool_call in response.tool_calls:
                    executed_call, result_payload = await self._execute_tool_with_reprompt(tool_call)

                    from ..tools.formatter import format_tool_result
                    formatted_result = format_tool_result(executed_call.name, result_payload.get("result") if isinstance(result_payload, dict) else result_payload)

                    self._record_message(
                        MessageRole.TOOL,
                        formatted_result,
                        tool_call_id=executed_call.id,
                        name=executed_call.name,
                    )

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


class ToolHandlerMixin:
    """Mixin for tool execution and definition management."""
    llm_provider: Optional[LLMProvider]
    tool_discovery: ToolDiscovery
    message_builder: MessageBuilder
    response_cache: Optional[LLMResponseCache]
    mcp_router: MCPRouter
    tool_error_reprompt_skill: ToolErrorRepromptSkill

    async def _refresh_tool_definitions(self: Any) -> None:
        supports_tools = bool(self.llm_provider and self.llm_provider.supports_tools)
        tool_defs_changed = await self.tool_discovery.refresh_async(supports_tools)

        if tool_defs_changed:
            self._tool_definitions = self.tool_discovery.tools
            self.message_builder.tool_definitions = self._tool_definitions
            if self.response_cache:
                self.response_cache.invalidate_all()

    def _refresh_tool_definitions_sync(self: Any) -> None:
        supports_tools = bool(self.llm_provider and self.llm_provider.supports_tools)
        tool_defs_changed = self.tool_discovery.refresh_sync(supports_tools)

        if tool_defs_changed:
            self._tool_definitions = self.tool_discovery.tools
            self.message_builder.tool_definitions = self._tool_definitions
            if self.response_cache:
                self.response_cache.invalidate_all()

    async def _execute_tool_with_reprompt(
        self: Any,
        tool_call: ProviderToolCall,
    ) -> tuple[ProviderToolCall, dict[str, Any]]:
        current_call = tool_call
        attempt = 0
        tools_payload = self._get_tools_payload()

        while True:
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


class CacheHandlerMixin:
    """Mixin for response caching logic."""
    response_cache: Optional[LLMResponseCache]
    llm_provider: Optional[LLMProvider]
    session_id: str
    llm_config: LLMConfig
    config: AgentConfig

    def _response_cache_lookup(
        self: Any,
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
        self: Any,
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
