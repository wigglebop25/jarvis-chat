import logging
from typing import Optional
from pathlib import Path

from .config import AgentConfig
from .llm import create_provider
from .mcp.multi_endpoint_router import MultiEndpointMCPRouter
from .rag.retriever import get_rag_retriever
from .context_cache import SessionContextCache
from .response_cache import LLMResponseCache
from .langgraph_agent import LangGraphChatAgent

logger = logging.getLogger(__name__)

class ChatAgentBuilder:
    """Builder for assembling ChatAgent instances with all dependencies."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.llm_config = self.config.llm
        
    def with_llm(self, provider: str, model: str) -> "ChatAgentBuilder":
        self.llm_config.provider = provider
        self.llm_config.model = model
        return self
        
    def build_langgraph_agent(self) -> LangGraphChatAgent:
        """Build the modern LangGraph-based agent."""
        # 1. Initialize LLM Provider
        llm_provider = create_provider(
            self.llm_config.provider, 
            **self.llm_config.get_provider_kwargs()
        )
        
        # 2. Initialize MCP Router
        mcp_router = MultiEndpointMCPRouter(
            system_endpoint=self.config.mcp.system_url,
            spotify_endpoint=self.config.mcp.spotify_url,
            system_transport=self.config.mcp.system_transport,
            spotify_transport=self.config.mcp.spotify_transport,
        )
        
        # 3. Initialize RAG
        rag_retriever = get_rag_retriever()
        
        # 4. Initialize Caches
        context_cache = None
        if self.config.context_cache_enabled:
            path = Path(self.config.context_cache_path).expanduser() if self.config.context_cache_path else None
            context_cache = SessionContextCache(
                llm_provider=llm_provider,
                requested_dtype=self.config.context_dtype,
                max_turns=self.config.context_cache_max_turns,
                summary_keep_last=self.config.context_cache_summary_keep_last,
                token_budget=self.config.context_token_budget,
                persistence_path=path,
            )
            
        response_cache = None
        if self.config.llm_response_cache_enabled:
            path = Path(self.config.llm_response_cache_path).expanduser() if self.config.llm_response_cache_path else None
            response_cache = LLMResponseCache(
                ttl_seconds=self.config.llm_response_cache_ttl_seconds,
                max_entries=self.config.llm_response_cache_max_entries,
                min_chars=self.config.llm_response_cache_min_chars,
                persistence_path=path,
            )
            
        # 5. Assemble Agent
        return LangGraphChatAgent(
            config=self.config,
            llm_config=self.llm_config,
            llm_provider=llm_provider,
            mcp_router=mcp_router,
            rag_retriever=rag_retriever,
            response_cache=response_cache,
            context_cache=context_cache,
        )
