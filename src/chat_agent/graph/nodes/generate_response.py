"""Generate LLM response."""

import time
import logging
from typing import Any, Dict, Optional
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ...llm.base import LLMProvider
from ...response_cache import LLMResponseCache
from ...config import LLMConfig
from ..state import AgentState
from .helpers import get_context_filtered_messages, cache_lookup, sanitize_assistant_text

logger = logging.getLogger(__name__)


async def generate_response_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that calls the LLM provider to generate a response.
    
    In the ReAct framework, this node serves two purposes:
    1. Reasoning: The LLM analyzes the current state and decides on a tool call.
    2. Final Response: After tool observations are in state, the LLM generates the final answer.
    """
    logger.info("[LangGraph] Entering generate_response_node")
    configurable = config.get("configurable", {})
    llm_provider: Optional[LLMProvider] = configurable.get("llm_provider")
    tools_payload = configurable.get("tools_payload")
    response_cache: Optional[LLMResponseCache] = configurable.get("response_cache")
    llm_config: Optional[LLMConfig] = configurable.get("llm_config")
    
    if not llm_provider:
        logger.error("[LangGraph] LLM provider not found in config")
        raise ValueError("LLM provider not found in config")
        
    # Cache Lookup
    logger.info("[LangGraph] Checking cache...")
    cached_text, key_bundle = cache_lookup(state, config)
    if cached_text is not None:
        logger.info("[LangGraph] Cache hit!")
        return {"messages": [AIMessage(content=cached_text)], "usage": {"cache_hit": True}}
        
    logger.info("[LangGraph] Preparing messages for LLM...")
    messages = get_context_filtered_messages(state, config)
    
    logger.info(f"[LangGraph] Calling LLM ({llm_config.provider if llm_config else 'unknown'})...")
    request_started = time.perf_counter()
    response = await llm_provider.complete(messages, tools=tools_payload)
    latency_ms = (time.perf_counter() - request_started) * 1000.0
    logger.info(f"[LangGraph] LLM response received in {latency_ms:.2f}ms")
    
    new_messages = []
    tool_calls = []
    
    # Check for execution loops using LangChain message types
    # This prevents the model from ignoring tool results and calling the same tool again.
    if response.tool_calls and len(state["messages"]) >= 2:
        last_msg = state["messages"][-1]
        prev_ai_msg = state["messages"][-2]
        
        # If the last thing in the graph was a tool result, and the one before that was the tool call
        if isinstance(last_msg, ToolMessage) and isinstance(prev_ai_msg, AIMessage):
            if prev_ai_msg.tool_calls:
                current_tc = response.tool_calls[0]
                prev_tc = prev_ai_msg.tool_calls[0]
                
                # Check if it's the exact same tool call
                prev_name = prev_tc.get("name") if isinstance(prev_tc, dict) else getattr(prev_tc, "name", "")
                prev_args = prev_tc.get("args") if isinstance(prev_tc, dict) else getattr(prev_tc, "arguments", {})
                if current_tc.name == prev_name and current_tc.arguments == (prev_args or {}):
                    logger.debug(f"[LangGraph] Execution loop detected for {current_tc.name}; returning last tool result.")
                    response.tool_calls = []
                    fallback_text = str(last_msg.content).strip() if last_msg.content else ""
                    response.text = fallback_text or "I have fetched the latest data. Please check the playlists or queue displayed above."

    if response.tool_calls:
        logger.info(f"[LangGraph] LLM requested {len(response.tool_calls)} tool calls")
        for tc in response.tool_calls:
            logger.info(f"[LangGraph]   Tool: {tc.name} (id: {tc.id})")
            tool_calls.append({
                "name": tc.name,
                "args": tc.arguments,
                "id": tc.id
            })
    else:
        logger.info("[LangGraph] No tool calls requested by LLM")
    
    sanitized_text = sanitize_assistant_text(response.text or "")
    ai_message = AIMessage(content=sanitized_text, tool_calls=tool_calls)
    new_messages.append(ai_message)
    
    # Cache Store
    if response_cache and key_bundle and llm_config and not tool_calls and sanitized_text:
        store_decision = response_cache.should_store_response(sanitized_text)
        if store_decision.cacheable:
            cache_key, transcript_key, tools_fingerprint = key_bundle
            response_cache.store(
                key=cache_key,
                response_text=sanitized_text,
                source_latency_ms=latency_ms,
                session_id=state["session_id"],
                provider=llm_config.provider,
                model=llm_config.model,
                transcript_key=transcript_key,
                tools_fingerprint=tools_fingerprint,
            )
        else:
            response_cache.record_skip(store_decision.reason)
        
    return {"messages": new_messages, "usage": response.usage}
