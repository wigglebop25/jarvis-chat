import logging
import json
from typing import Any, Optional, cast
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from .config import AgentConfig, LLMConfig
from .llm import create_provider
from .mcp import MCPRouter
from .graph.builder import create_graph
from .models import MessageRole, ChatMessage
from .tools.definitions import get_tool_definitions

logger = logging.getLogger(__name__)

class LangGraphChatAgent:
    """
    JARVIS Chat Agent implemented using LangGraph.
    
    This agent provides the same interface as the original ChatAgent
    but uses a state-based graph for orchestration.
    """

    def __init__(self, config: Optional[AgentConfig] = None, llm_config: Optional[LLMConfig] = None):
        self.config = config or AgentConfig()
        self.llm_config = llm_config or self.config.llm
        self.session_id = self.config.session_id
        
        # Initialize the graph
        self.graph = create_graph()
        
        # Internal state for manual history tracking
        self.messages: list[Any] = [SystemMessage(content=self.config.system_prompt)]
        
        self.llm_provider = None
        try:
            self.llm_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
        except Exception as e:
            logger.warning(f"Could not initialize LLM provider: {e}. Using fallback mode.")

        from .mcp.client import MCPClient
        mcp_client = MCPClient(base_url=self.config.mcp.url)
        self.mcp_router = MCPRouter(mcp_client=mcp_client)
        
        from .skills import ToolErrorRepromptSkill
        self.tool_error_reprompt_skill = ToolErrorRepromptSkill(
            max_retries=self.config.tool_retry_attempts,
            base_backoff_seconds=self.config.tool_retry_backoff_seconds,
        )
        
        self.response_cache = None
        if self.config.llm_response_cache_enabled:
            from .response_cache import LLMResponseCache
            from pathlib import Path
            path = Path(self.config.llm_response_cache_path).expanduser() if self.config.llm_response_cache_path else None
            self.response_cache = LLMResponseCache(
                ttl_seconds=self.config.llm_response_cache_ttl_seconds,
                max_entries=self.config.llm_response_cache_max_entries,
                min_chars=self.config.llm_response_cache_min_chars, persistence_path=path,
            )
        
        self.context_cache = None
        if self.config.context_cache_enabled and self.llm_provider:
            from .context_cache import SessionContextCache
            from pathlib import Path
            path = Path(self.config.context_cache_path).expanduser() if self.config.context_cache_path else None
            self.context_cache = SessionContextCache(
                llm_provider=self.llm_provider,
                requested_dtype=self.config.context_dtype, max_turns=self.config.context_cache_max_turns,
                summary_keep_last=self.config.context_cache_summary_keep_last,
                token_budget=self.config.context_token_budget, persistence_path=path,
            )

        self._tool_definitions = get_tool_definitions()

    async def process_transcript(self, transcript: str) -> str:
        if not transcript or not transcript.strip():
            return "I didn't catch that. Could you please repeat?"

        if self.config.log_transcripts:
            logger.info(f"Processing transcript (LangGraph): {transcript}")

        from .graph.state import AgentState
        # Prepare inputs for the graph
        # LangGraph handles history via checkpointer
        inputs: AgentState = {
            "messages": [HumanMessage(content=transcript)],
            "session_id": self.session_id,
            "intent": None,
            "execution_error": None,
            "last_tool_call_id": None
        }
        
        # Prepare config for nodes (passing dependencies and thread_id)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": self.session_id,
                "llm_provider": self.llm_provider,
                "mcp_router": self.mcp_router,
                "reprompt_skill": self.tool_error_reprompt_skill,
                "response_cache": self.response_cache,
                "context_cache": self.context_cache,
                "agent_config": self.config,
                "llm_config": self.llm_config,
                "session_id": self.session_id,
                "tools_payload": self._tool_definitions if self.llm_provider and self.llm_provider.supports_tools else None
            }
        }
        
        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(inputs, config=config)
            
            # Update local message history from the graph's state
            self.messages = final_state["messages"]
            
            # Find the latest AIMessage that has content
            for m in reversed(self.messages):
                if isinstance(m, AIMessage) and m.content:
                    return cast(str, m.content)
            
            return "Task completed."
            
        except Exception as e:
            logger.error(f"LangGraph execution error: {e}")
            return f"I encountered an error while processing your request: {e}"

    def clear_context(self) -> None:
        self.messages = [SystemMessage(content=self.config.system_prompt)]
        if self.context_cache:
            self.context_cache.clear_session(self.session_id)
        if self.response_cache:
            self.response_cache.invalidate_session(self.session_id)

    def change_model(self, model_name: str, provider_name: Optional[str] = None) -> None:
        if provider_name:
            self.llm_config.provider = provider_name
        self.llm_config.model = model_name
        self.llm_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
        if self.context_cache:
            self.context_cache.update_provider(self.llm_provider)

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id.strip() if session_id else "default"

    def get_conversation_history(self) -> list:
        history = []
        for m in self.messages:
            role = MessageRole.ASSISTANT # Default
            if isinstance(m, HumanMessage): 
                role = MessageRole.USER
            elif isinstance(m, SystemMessage): 
                role = MessageRole.SYSTEM
            elif isinstance(m, AIMessage): 
                role = MessageRole.ASSISTANT
            elif isinstance(m, ToolMessage): 
                role = MessageRole.TOOL
            
            content = cast(str, m.content)
            if isinstance(m, AIMessage) and m.tool_calls:
                content += f"\n[Tool Calls: {json.dumps(m.tool_calls)}]"

            history.append(ChatMessage(role=role, content=content))
        return history

    def get_cache_stats(self) -> dict[str, Any]:
        stats = {"enabled": self.response_cache is not None}
        if self.response_cache:
            stats.update(self.response_cache.get_stats())
        if self.context_cache:
            stats.update(self.context_cache.get_stats(self.session_id))
        return stats

    def register_context_artifact(self, name: str, values: list[float], source_dtype: str = "fp32") -> dict:
        if not self.context_cache: raise RuntimeError("Context cache disabled.")
        return self.context_cache.register_artifact(self.session_id, name, values, source_dtype).to_dict()

    def convert_context_artifact_dtype(self, name: str, target_dtype: str) -> dict:
        if not self.context_cache: raise RuntimeError("Context cache disabled.")
        return self.context_cache.convert_artifact_dtype(self.session_id, name, target_dtype).to_dict()
