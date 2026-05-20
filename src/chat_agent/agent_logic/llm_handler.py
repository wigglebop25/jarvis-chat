"""
agent_logic/llm_handler.py
──────────────────────────
LLMHandlerMixin: async and sync LLM completion with response caching.
"""
from __future__ import annotations
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

from ..models import Intent, MessageRole

if TYPE_CHECKING:
    from ..llm import LLMProvider
    from ..config import AgentConfig, LLMConfig
    from ..response_cache import LLMResponseCache

logger = logging.getLogger(__name__)


class LLMHandlerMixin:
    """Mixin for LLM interaction logic."""
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
                self.last_usage = response.usage
                self._record_message(MessageRole.ASSISTANT, response.text)
                self._response_cache_store(
                    key_bundle=key_bundle,
                    response_text=response.text,
                    latency_ms=latency_ms,
                    used_tools=False,
                )
                return response.text
            self.last_usage = response.usage
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
                self.last_usage = {"cache_hit": True}
                self._record_message(MessageRole.ASSISTANT, cached_text)
                return cached_text

            request_started = time.perf_counter()
            if not self.llm_provider:
                raise RuntimeError("LLM provider not initialized for async completion")
            response = await self.llm_provider.complete(messages, tools=tools_payload)
            latency_ms = (time.perf_counter() - request_started) * 1000.0

            self.last_usage = response.usage

            latest_text = ""
            used_tools = bool(response.tool_calls)

            if response.tool_calls:
                latest_text = ""
            elif response.text:
                latest_text = response.text
                self._record_message(MessageRole.ASSISTANT, response.text)

            while response.tool_calls:
                used_tools = True
                latest_formatted_results = []
                has_error = False

                for tool_call in response.tool_calls:
                    executed_call, result_payload = await self._execute_tool_with_reprompt(tool_call)

                    from ..tools.formatter import format_tool_result, format_tool_error

                    if result_payload.get("is_error"):
                        has_error = True
                        formatted_result = format_tool_error(
                            executed_call.name,
                            str(result_payload.get("error") or "Unknown error"),
                        )
                    else:
                        formatted_result = format_tool_result(
                            executed_call.name,
                            result_payload.get("result") if isinstance(result_payload, dict) else result_payload,
                        )

                    self._record_message(
                        MessageRole.TOOL,
                        formatted_result,
                        tool_call_id=executed_call.id,
                        name=executed_call.name,
                    )
                    latest_formatted_results.append(formatted_result)

                # EARLY EXIT: tool succeeded — don't ask LLM to summarize
                if not has_error:
                    final_text = "\n".join(latest_formatted_results)
                    self._record_message(MessageRole.ASSISTANT, final_text)
                    self._response_cache_store(
                        key_bundle=key_bundle,
                        response_text=final_text,
                        latency_ms=latency_ms,
                        used_tools=used_tools,
                    )
                    return final_text

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
