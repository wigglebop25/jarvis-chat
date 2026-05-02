"""
Cache eligibility guardrails.

Evaluates whether transcripts and responses are safe to cache
based on content patterns and context.
"""

from dataclasses import dataclass
import re
from typing import Any


# Content patterns for cache eligibility checks
FRESHNESS_HINT_PATTERN = re.compile(
    r"\b(now|today|current|currently|latest|recent|real[- ]?time|time|date|status|weather|forecast|news)\b",
    re.IGNORECASE,
)
ACTION_PREFIX_PATTERN = re.compile(
    r"^\s*(turn|set|toggle|enable|disable|mute|unmute|play|pause|stop|list|show|open|close|delete|move|organize)\b",
    re.IGNORECASE,
)
CONTROL_ENTITY_PATTERN = re.compile(
    r"\b(volume|spotify|music|wifi|wi-fi|bluetooth|network|cpu|ram|memory|storage|disk|drive|folder|file|directory)\b",
    re.IGNORECASE,
)
PATH_HINT_PATTERN = re.compile(r"([a-zA-Z]:[\\/]|\\\\)")
CONTEXT_DEPENDENT_PATTERN = re.compile(
    r"\b(this|that|it|they|those|these|above|earlier|previous|before|last answer|conversation|history)\b",
    re.IGNORECASE,
)
TOOL_REPROMPT_HINT = "Tool execution failed."


@dataclass(slots=True)
class CacheEligibilityDecision:
    """Result of cache eligibility evaluation."""
    cacheable: bool
    reason: str


def evaluate_transcript_eligibility(
    *,
    transcript: str,
    intent_type: str | None,
    supports_tools: bool,
    tools_payload: list[dict[str, Any]] | None,
    allow_tool_providers: bool,
    messages: list[dict[str, str]],
) -> CacheEligibilityDecision:
    """
    Evaluate if a transcript is eligible for caching.
    
    Args:
        transcript: User transcript/query
        intent_type: Recognized intent type (if any)
        supports_tools: Whether provider supports tool calling
        tools_payload: Available tool definitions
        allow_tool_providers: Whether to cache with tool providers
        messages: Conversation message history
    
    Returns:
        CacheEligibilityDecision with cacheable flag and reason
    """
    cleaned = (transcript or "").strip()
    
    if not cleaned:
        return CacheEligibilityDecision(False, "empty_transcript")
    
    if intent_type and intent_type != "general_query":
        return CacheEligibilityDecision(False, "intent_not_general_query")
    
    if supports_tools and tools_payload and not allow_tool_providers:
        return CacheEligibilityDecision(False, "tool_provider_disabled")
    
    if FRESHNESS_HINT_PATTERN.search(cleaned):
        return CacheEligibilityDecision(False, "freshness_sensitive_query")
    
    if ACTION_PREFIX_PATTERN.search(cleaned):
        return CacheEligibilityDecision(False, "action_like_prompt")
    
    if CONTROL_ENTITY_PATTERN.search(cleaned):
        return CacheEligibilityDecision(False, "stateful_entity_prompt")
    
    if PATH_HINT_PATTERN.search(cleaned):
        return CacheEligibilityDecision(False, "filesystem_prompt")
    
    if CONTEXT_DEPENDENT_PATTERN.search(cleaned):
        return CacheEligibilityDecision(False, "context_dependent_prompt")
    
    # Check context messages for tool usage
    if any((msg.get("role") == "tool") for msg in messages):
        return CacheEligibilityDecision(False, "tool_context_present")
    
    # Check recent messages for tool retry context
    if any(
        (msg.get("role") == "system" and TOOL_REPROMPT_HINT in msg.get("content", ""))
        for msg in messages[-8:]
    ):
        return CacheEligibilityDecision(False, "tool_retry_context")
    
    return CacheEligibilityDecision(True, "cacheable")


def evaluate_response_eligibility(
    response_text: str,
    min_chars: int = 24,
) -> CacheEligibilityDecision:
    """
    Evaluate if a response is eligible for caching.
    
    Args:
        response_text: LLM response text
        min_chars: Minimum response length to cache
    
    Returns:
        CacheEligibilityDecision with storeable flag and reason
    """
    cleaned = (response_text or "").strip()
    
    if not cleaned:
        return CacheEligibilityDecision(False, "empty_response")
    
    if len(cleaned) < min_chars:
        return CacheEligibilityDecision(False, "response_too_short")
    
    lowered = cleaned.lower()
    if lowered.startswith("i couldn't") or "unknown error" in lowered:
        return CacheEligibilityDecision(False, "error_response")
    
    return CacheEligibilityDecision(True, "storeable")
