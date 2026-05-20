"""Shared helper functions for graph nodes."""

import re
from typing import Any, Dict, Tuple, Optional, cast
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ...models import MessageRole
from ...response_cache import LLMResponseCache
from ...context_cache import SessionContextCache
from ...config import AgentConfig, LLMConfig
from ...llm.base import LLMProvider
from ..state import AgentState

REASONING_LEAKAGE_MARKERS = (
    "the user is asking",
    "the user wants",
    "i should",
    "i need to",
    "looking back at",
)


def sanitize_assistant_text(text: str) -> str:
    """Remove reasoning leakage from assistant text."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    cleaned = re.sub(
        r"--- TOOL EXECUTION RESULT ---.*?-----------------------------",
        "",
        cleaned,
        flags=re.DOTALL,
    ).strip()

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return "I completed that step, but couldn't produce a clean final message. Please rephrase the request."

    lowered = cleaned.lower()
    if not cleaned or not any(marker in lowered for marker in REASONING_LEAKAGE_MARKERS):
        return cleaned

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return cleaned
    return lines[-1]


def map_to_provider_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Map LangChain messages to the provider format."""
    provider_messages: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            provider_messages.append({"role": MessageRole.USER.value, "content": cast(str, m.content)})
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": MessageRole.ASSISTANT.value, "content": m.content or ""}
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg["tool_calls"] = [
                    {"id": tc["id"], "name": tc["name"], "arguments": tc["args"]}
                    for tc in m.tool_calls
                ]
            provider_messages.append(msg)
        elif isinstance(m, SystemMessage):
            provider_messages.append({"role": MessageRole.SYSTEM.value, "content": cast(str, m.content)})
        elif isinstance(m, ToolMessage):
            provider_messages.append({
                "role": MessageRole.TOOL.value,
                "content": cast(str, m.content),
                "tool_call_id": m.tool_call_id,
                "name": cast(str, m.name)
            })
    return provider_messages


def get_context_filtered_messages(state: AgentState, config: RunnableConfig) -> list[Dict[str, str]]:
    """Helper to get filtered messages using context_cache if available."""
    configurable = config.get("configurable", {})
    context_cache: Optional[SessionContextCache] = configurable.get("context_cache")
    session_id = state.get("session_id", "default")
    agent_config: Optional[AgentConfig] = configurable.get("agent_config")
    system_prompt = agent_config.system_prompt if agent_config else ""
    
    # Inject RAG context if available
    rag_context = state.get("rag_context")
    if rag_context:
        # Use delimiters for better RAG integration
        system_prompt = f"{system_prompt}\n\n[RETRIEVED CONTEXT]\n{rag_context}\n[END CONTEXT]"

    if context_cache:
        # Sync latest state to cache
        context_cache.clear_session(session_id)
        for m in state["messages"]:
            role = "assistant"
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, SystemMessage):
                role = "system"
                continue # Skip existing system messages as we provide a new one
            elif isinstance(m, ToolMessage):
                role = "tool"
            content = cast(str, m.content)
            context_cache.add_message(session_id, role, content)
            
        return context_cache.build_messages(session_id, system_prompt)
    
    # Ensure system prompt is at the start of messages if not using context_cache
    messages = map_to_provider_messages(state["messages"])
    
    # Filter out any existing system messages to avoid duplicates
    messages = [m for m in messages if m.get("role") != MessageRole.SYSTEM.value]
    
    # Prepend the updated system prompt
    return [{"role": MessageRole.SYSTEM.value, "content": system_prompt}] + messages


def cache_lookup(state: AgentState, config: RunnableConfig) -> Tuple[Optional[str], Optional[Tuple[str, str, str]]]:
    """Helper for response cache lookup."""
    configurable = config.get("configurable", {})
    response_cache: Optional[LLMResponseCache] = configurable.get("response_cache")
    llm_provider: Optional[LLMProvider] = configurable.get("llm_provider")
    agent_config: Optional[AgentConfig] = configurable.get("agent_config")
    llm_config: Optional[LLMConfig] = configurable.get("llm_config")
    
    if not response_cache or not llm_provider or not agent_config or not llm_config:
        return None, None
        
    messages = get_context_filtered_messages(state, config)
    # Find last user message as transcript
    transcript = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            transcript = cast(str, m.content)
            break
            
    intent = state.get("intent")
    intent_type = intent.get("type") if intent else None
    tools_payload = configurable.get("tools_payload")
    
    decision = response_cache.evaluate_eligibility(
        transcript=transcript,
        intent_type=intent_type,
        supports_tools=llm_provider.supports_tools,
        tools_payload=tools_payload,
        allow_tool_providers=agent_config.llm_response_cache_allow_tool_providers,
        messages=messages,
    )
    
    if not decision.cacheable:
        response_cache.record_skip(decision.reason)
        return None, None

    key_bundle = response_cache.build_key(
        transcript=transcript,
        session_id=state["session_id"],
        provider=llm_config.provider,
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        system_prompt=agent_config.system_prompt,
        tools_payload=tools_payload,
    )
    cached_text = response_cache.lookup(key_bundle[0])
    return cached_text, key_bundle


def get_transcript_from_state(state: AgentState) -> str:
    """Extract transcript from the last human message in state."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return cast(str, m.content)
    return ""
