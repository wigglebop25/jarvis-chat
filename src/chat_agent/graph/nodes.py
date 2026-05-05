import time
from typing import Any, Dict, Optional, Tuple, cast
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from ..llm.base import LLMProvider
from ..mcp import MCPRouter
from ..models import MessageRole
from ..skills import ToolErrorRepromptSkill
from ..response_cache import LLMResponseCache
from ..context_cache import SessionContextCache
from ..config import AgentConfig, LLMConfig

def _map_to_provider_messages(messages: list[Any]) -> list[Dict[str, str]]:
    """Map LangChain messages to the provider format."""
    provider_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            provider_messages.append({"role": MessageRole.USER.value, "content": cast(str, m.content)})
        elif isinstance(m, AIMessage):
            msg: Dict[str, Any] = {"role": MessageRole.ASSISTANT.value, "content": m.content or ""}
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
                 "name": m.name
             })
    return provider_messages

def _get_context_filtered_messages(state: AgentState, config: RunnableConfig) -> list[Dict[str, str]]:
    """Helper to get filtered messages using context_cache if available."""
    configurable = config.get("configurable", {})
    context_cache: Optional[SessionContextCache] = configurable.get("context_cache")
    session_id = state.get("session_id", "default")
    agent_config: Optional[AgentConfig] = configurable.get("agent_config")
    system_prompt = agent_config.system_prompt if agent_config else ""

    if context_cache:
        # Sync latest state to cache
        context_cache.clear_session(session_id)
        for m in state["messages"]:
            role = "assistant"
            if isinstance(m, HumanMessage): role = "user"
            elif isinstance(m, SystemMessage): role = "system"
            elif isinstance(m, ToolMessage): role = "tool"
            content = cast(str, m.content)
            context_cache.add_message(session_id, role, content)
            
        return context_cache.build_messages(session_id, system_prompt)
    
    return _map_to_provider_messages(state["messages"])

def _cache_lookup(state: AgentState, config: RunnableConfig) -> Tuple[Optional[str], Optional[Tuple[str, str, str]]]:
    """Helper for response cache lookup."""
    configurable = config.get("configurable", {})
    response_cache: Optional[LLMResponseCache] = configurable.get("response_cache")
    llm_provider: Optional[LLMProvider] = configurable.get("llm_provider")
    agent_config: Optional[AgentConfig] = configurable.get("agent_config")
    llm_config: Optional[LLMConfig] = configurable.get("llm_config")
    
    if not response_cache or not llm_provider or not agent_config or not llm_config:
        return None, None
        
    messages = _get_context_filtered_messages(state, config)
    # Find last user message as transcript
    transcript = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            transcript = cast(str, m.content)
            break
            
    intent = state.get("intent")
    tools_payload = configurable.get("tools_payload")
    
    decision = response_cache.evaluate_eligibility(
        transcript=transcript,
        intent_type=intent.type.value if intent else None,
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

async def call_model(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that calls the LLM provider.
    """
    configurable = config.get("configurable", {})
    llm_provider: Optional[LLMProvider] = configurable.get("llm_provider")
    tools_payload = configurable.get("tools_payload")
    response_cache: Optional[LLMResponseCache] = configurable.get("response_cache")
    llm_config: Optional[LLMConfig] = configurable.get("llm_config")
    
    if not llm_provider:
        raise ValueError("LLM provider not found in config")
        
    # Cache Lookup
    cached_text, key_bundle = _cache_lookup(state, config)
    if cached_text is not None:
        return {"messages": [AIMessage(content=cached_text)]}
        
    messages = _get_context_filtered_messages(state, config)
    
    request_started = time.perf_counter()
    response = await llm_provider.complete(messages, tools=tools_payload)
    latency_ms = (time.perf_counter() - request_started) * 1000.0
    
    new_messages = []
    tool_calls = []
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append({
                "name": tc.name,
                "args": tc.arguments,
                "id": tc.id
            })
    
    ai_message = AIMessage(content=response.text or "", tool_calls=tool_calls)
    new_messages.append(ai_message)
    
    # Cache Store
    if response_cache and key_bundle and llm_config and not tool_calls and response.text:
        store_decision = response_cache.should_store_response(response.text)
        if store_decision.cacheable:
            cache_key, transcript_key, tools_fingerprint = key_bundle
            response_cache.store(
                key=cache_key,
                response_text=response.text,
                source_latency_ms=latency_ms,
                session_id=state["session_id"],
                provider=llm_config.provider,
                model=llm_config.model,
                transcript_key=transcript_key,
                tools_fingerprint=tools_fingerprint,
            )
        else:
            response_cache.record_skip(store_decision.reason)
        
    return {"messages": new_messages}

async def execute_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that executes tool calls.
    """
    configurable = config.get("configurable", {})
    mcp_router: Optional[MCPRouter] = configurable.get("mcp_router")
    
    if not mcp_router:
        raise ValueError("MCP router not found in config")
        
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}
        
    new_messages = []
    execution_error = None
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        
        # Execute the tool
        result = await mcp_router.execute_tool(tool_name, **tool_args)
        
        # Format the result
        from ..tools.formatter import format_tool_result, format_tool_error
        
        if result.is_error:
            content = format_tool_error(tool_name, result.error or "Unknown error")
            execution_error = result.error or "Unknown tool error"
        else:
            content = format_tool_result(tool_name, result.result)
            
        new_messages.append(ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name
        ))
        
    return {
        "messages": new_messages,
        "execution_error": execution_error
    }

async def handle_tool_error(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node that generates a reprompt for failed tool calls.
    """
    if not state.get("execution_error"):
        return {}
        
    configurable = config.get("configurable", {})
    reprompt_skill: Optional[ToolErrorRepromptSkill] = configurable.get("reprompt_skill")
    if not reprompt_skill:
        return {}
        
    # Find the failed tool call in history
    last_ai_message = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.tool_calls:
            last_ai_message = m
            break
            
    if not last_ai_message or not last_ai_message.tool_calls:
        return {}
        
    # For simplicity, we reprompt for the first tool call that failed
    failed_call = last_ai_message.tool_calls[0]
    
    reprompt = reprompt_skill.build_reprompt(
        failed_tool_name=failed_call["name"],
        failed_arguments=failed_call["args"],
        error_message=state.get("execution_error") or "Unknown error",
        attempt=1 # For now just 1, could track in state
    )
    
    return {
        "messages": [SystemMessage(content=reprompt)],
        "execution_error": None # Clear the error after reprompting
    }
