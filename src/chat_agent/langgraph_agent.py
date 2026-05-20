import logging
import json
from typing import Any, Optional, cast
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from .config import AgentConfig, LLMConfig
from .llm import create_provider
from .mcp import MCPRouter
from .mcp.multi_endpoint_router import MultiEndpointMCPRouter
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

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        llm_provider: Optional[Any] = None,
        mcp_router: Optional[Any] = None,
        rag_retriever: Optional[Any] = None,
        response_cache: Optional[Any] = None,
        context_cache: Optional[Any] = None,
    ):
        self.config = config or AgentConfig()
        self.llm_config = llm_config or self.config.llm
        self.session_id = self.config.session_id
        
        # Initialize the graph
        self.graph = create_graph()
        
        # Internal state for manual history tracking
        self.messages: list[Any] = [SystemMessage(content=self.config.system_prompt)]
        
        self.llm_provider = llm_provider
        if self.llm_provider is None:
            try:
                self.llm_provider = create_provider(self.llm_config.provider, **self.llm_config.get_provider_kwargs())
            except Exception as e:
                logger.warning(f"Could not initialize LLM provider: {e}. Using fallback mode.")

        self.mcp_router = mcp_router
        if self.mcp_router is None:
            if self.config.mcp.multi_endpoint_enabled:
                self.mcp_router = MultiEndpointMCPRouter(
                    system_endpoint=self.config.mcp.system_url,
                    spotify_endpoint=self.config.mcp.spotify_url,
                    system_transport=self.config.mcp.system_transport,
                    spotify_transport=self.config.mcp.spotify_transport,
                )
            else:
                from .mcp.client import MCPClient
                mcp_client = MCPClient(base_url=self.config.mcp.url)
                self.mcp_router = MCPRouter(mcp_client=mcp_client)
        
        from .rag.retriever import get_rag_retriever
        self.rag_retriever = rag_retriever or get_rag_retriever()
        
        from .skills import ToolErrorRepromptSkill
        self.tool_error_reprompt_skill = ToolErrorRepromptSkill(
            max_retries=self.config.tool_retry_attempts,
            base_backoff_seconds=self.config.tool_retry_backoff_seconds,
        )
        
        self.response_cache = response_cache
        if self.response_cache is None and self.config.llm_response_cache_enabled:
            from .response_cache import LLMResponseCache
            from pathlib import Path
            path = Path(self.config.llm_response_cache_path).expanduser() if self.config.llm_response_cache_path else None
            self.response_cache = LLMResponseCache(
                ttl_seconds=self.config.llm_response_cache_ttl_seconds,
                max_entries=self.config.llm_response_cache_max_entries,
                min_chars=self.config.llm_response_cache_min_chars, persistence_path=path,
            )
        
        self.context_cache = context_cache
        if self.context_cache is None and self.config.context_cache_enabled and self.llm_provider:
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
        self.last_usage: dict[str, Any] = {}

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
            "last_tool_call_id": None,
            "fast_path_executed": None,
            "usage": None,
            "rag_context": None
        }
        
        # Refresh tool definitions from MCP router (dynamic discovery)
        if self.mcp_router:
            try:
                raw_tools = await self.mcp_router.list_tools()
                from .tools.definitions import normalize_mcp_tool_definitions
                self._tool_definitions = normalize_mcp_tool_definitions(raw_tools)
            except Exception as e:
                logger.warning(f"Dynamic tool discovery failed: {e}. Using cached definitions.")

        # Prepare config for nodes (passing dependencies and thread_id)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": self.session_id,
                "llm_provider": self.llm_provider,
                "mcp_router": self.mcp_router,
                "rag_retriever": self.rag_retriever,
                "reprompt_skill": self.tool_error_reprompt_skill,
                "response_cache": self.response_cache,
                "context_cache": self.context_cache,
                "agent_config": self.config,
                "llm_config": self.llm_config,
                "session_id": self.session_id,
                "tools_payload": self._tool_definitions
            }
        }
        
        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(inputs, config=config)
            
            # Update local message history from the graph's state
            self.messages = final_state["messages"]
            self.last_usage = final_state.get("usage", {})
            
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

    def get_available_models(self) -> list:
        if self.llm_provider is None:
            return []
        return self.llm_provider.get_available_models()

    def get_available_models_detailed(self) -> list[dict[str, Any]]:
        """Fetch available models with detailed metadata."""
        if self.llm_provider is None:
            return []
        if hasattr(self.llm_provider, "get_available_models_detailed"):
            return self.llm_provider.get_available_models_detailed()
        return [{"name": m, "input_token_limit": 0} for m in self.llm_provider.get_available_models()]

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

    def get_mcp_server_count(self) -> int:
        """Get the number of configured MCP servers."""
        # Count configured endpoints in MultiEndpointMCPRouter
        if hasattr(self.mcp_router, 'mcp_client'):
            # MultiEndpointMCPRouter has mcp_client with system_endpoint and spotify_endpoint
            client = getattr(self.mcp_router, 'mcp_client', None)
            count = 0
            if client and getattr(client, 'system_endpoint', None):
                count += 1
            if client and getattr(client, 'spotify_endpoint', None):
                count += 1
            return count
        
        # Fallback for single-endpoint router
        return 1

    def register_context_artifact(self, name: str, values: list[float], source_dtype: str = "fp32") -> dict:
        if not self.context_cache:
            raise RuntimeError("Context cache disabled.")
        return self.context_cache.register_artifact(self.session_id, name, values, source_dtype).to_dict()

    def convert_context_artifact_dtype(self, name: str, target_dtype: str) -> dict:
        if not self.context_cache:
            raise RuntimeError("Context cache disabled.")
        return self.context_cache.convert_artifact_dtype(self.session_id, name, target_dtype).to_dict()
