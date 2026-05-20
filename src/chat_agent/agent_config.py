"""Agent-specific configuration."""

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .context_dtype import SUPPORTED_CONTEXT_DTYPES
from .llm_config import LLMConfig, OpenAIConfig
from .mcp_config import MCPConfig
from .cache_config import CacheConfig

# Default prompt file — sits next to this module
_DEFAULT_PROMPT_FILE = Path(__file__).parent / "system_prompt.md"
_FALLBACK_PROMPT = "You are JARVIS, an intelligent AI assistant. Be concise and helpful."


def _load_system_prompt() -> str:
    """Load system prompt from env-override path, local file, or inline fallback."""
    env_path = os.getenv("JARVIS_SYSTEM_PROMPT_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path).read_text(encoding="utf-8").strip()
    if _DEFAULT_PROMPT_FILE.exists():
        return _DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return _FALLBACK_PROMPT


class AgentConfig(BaseModel):
    """Main Chat Agent configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    system_prompt: str = Field(default_factory=_load_system_prompt)

    # User-facing fallback when the LLM fails entirely
    fallback_message: str = Field(
        default="I'm not sure how to help with that. Try rephrasing or type ? for help."
    )

    debug: bool = Field(default=False)
    log_transcripts: bool = Field(default=True)
    session_id: str = Field(default_factory=lambda: os.getenv("CHAT_SESSION_ID", "default"))
    agent_type: str = Field(
        default_factory=lambda: os.getenv("AGENT_TYPE", "langgraph"),
        description="Agent implementation type: 'legacy' or 'langgraph'"
    )
    context_dtype: str = Field(default_factory=lambda: os.getenv("CONTEXT_DTYPE", "fp16"))
    tool_retry_attempts: int = Field(default=2, ge=0)
    tool_retry_backoff_seconds: float = Field(default=0.5, ge=0.0)

    # Context cache configuration
    context_cache_enabled: bool = Field(default=True, description="Enable per-session context cache and compaction")
    context_cache_path: str | None = Field(default=None, description="Path for persisting per-session context cache (JSON)")
    context_cache_max_turns: int = Field(default=50, description="Maximum number of recent turns to keep before compaction")
    context_cache_summary_keep_last: int = Field(default=2, description="Number of last summaries to keep when compacting context")
    context_token_budget: int = Field(default=1024, description="Token budget to use when building context for prompts")

    # LLM response cache configuration
    llm_response_cache_enabled: bool = Field(default=False, description="Enable caching of LLM responses to save latency and tokens")
    llm_response_cache_path: str | None = Field(default=None, description="Persistence path for response cache")
    llm_response_cache_ttl_seconds: int = Field(default=180, description="TTL for cached LLM responses in seconds")
    llm_response_cache_max_entries: int = Field(default=256, description="Max number of cache entries")
    llm_response_cache_min_chars: int = Field(default=24, description="Minimum response length to consider caching")

    # Whether to allow response caching for LLMs that support tool providers
    llm_response_cache_allow_tool_providers: bool = Field(default=False, description="Allow caching when LLM supports tools and tools are present")

    @field_validator("context_dtype")
    @classmethod
    def validate_context_dtype(cls, v: str) -> str:
        normalized = v.lower().strip()
        if normalized not in SUPPORTED_CONTEXT_DTYPES:
            valid = sorted(SUPPORTED_CONTEXT_DTYPES)
            raise ValueError(f"context_dtype must be one of {valid}, got '{v}'")
        return normalized
