"""
agent_logic/cache_handler.py
──────────────────────────────
CacheHandlerMixin: LLM response cache lookup and storage.
"""
from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING

from ..models import Intent

if TYPE_CHECKING:
    from ..llm import LLMProvider
    from ..config import AgentConfig, LLMConfig
    from ..response_cache import LLMResponseCache


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
