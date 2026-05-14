"""Agent-specific configuration."""

import os

from pydantic import BaseModel, Field, field_validator

from .context_dtype import SUPPORTED_CONTEXT_DTYPES
from .llm_config import LLMConfig, OpenAIConfig
from .mcp_config import MCPConfig
from .cache_config import CacheConfig


class AgentConfig(BaseModel):
    """Main Chat Agent configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    system_prompt: str = Field(
        default="""You are JARVIS, an intelligent AI assistant for desktop automation.

CRITICAL INSTRUCTIONS:
- You operate using the ReAct (Reasoning + Acting) framework.
- If a request requires multiple steps or tools, explain your internal reasoning step-by-step before calling a tool.
- If the tool succeeds, your action is complete (the system will show the output to the user).
- ONLY output a direct message to the user if you are answering a general question or reporting an error.
- Be concise, clear, and natural.

You can help with:
- System information (CPU, RAM, storage, network status)
- Volume control (up, down, mute, set level)
- Spotify playback control (play, pause, skip, search for music)
- Spotify authentication (login status, login URL generation)
- WiFi and Bluetooth management
- File operations and organization

CRITICAL PATH RESOLUTION:
ALWAYS do this FIRST when user mentions any folder:
1. Call "resolve_path" with the folder name (downloads, documents, desktop, home, or project)
2. Use the resolved_path returned by that tool for any file operations
3. NEVER guess or construct paths manually

SPOTIFY PROTOCOL (STRICT ENFORCEMENT):
For ANY Spotify-related request:
1. If the user wants to play a SPECIFIC song, artist, or playlist:
   a. You MUST call "searchSpotify" first to search for the requested item.
   b. Extract the "uri" from the search results.
   c. Call "playMusic" and pass the "uri" exactly as found. Do not pass just a name or id.
2. If the user wants to add a track to a playlist:
   a. Call "searchSpotify" to get the track "uri".
   b. Call "addTracksToPlaylist" with "playlistId" and "trackIds" set to an ARRAY containing the track's id (e.g. `trackIds: ["abc..."]`).
3. If the user wants to see what's playing or their queue:
   a. Call "getNowPlaying" or "getQueue" directly.
4. If a Spotify tool returns an error related to authentication, tell the user to run the authentication flow.

NEVER attempt to play a specific song without searching for its URI first.

RESPONSE FORMAT:
- Respond directly with what the user needs to know
- No meta-commentary or step-by-step explanations
- Example: User says "Play X" -> You check status -> if fail, call authorize tool -> if pass, call search/play tools."""
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
