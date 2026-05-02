"""
Chat Agent Configuration

Configuration management for the JARVIS Chat Agent.
Loads settings from environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from .context_dtype import SUPPORTED_CONTEXT_DTYPES


load_dotenv()


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class LLMConfig(BaseModel):
    """Configuration for LLM provider selection and parameters."""
    
    provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"),
        description="LLM provider: 'ollama', 'openai', 'gemini', 'copilot'"
    )
    model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama3"),
        description="Model name (provider-specific)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    max_tokens: int = Field(
        default=2048,
        ge=100,
        description="Maximum tokens in response"
    )
    
    def get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for create_provider()."""
        kwargs: dict[str, Any] = {"model": self.model}
        
        # Only add provider-specific params
        if self.provider in ["openai", "gemini"]:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens
        
        return kwargs
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = {"ollama", "openai", "gemini", "copilot"}
        if v not in valid:
            raise ValueError(f"Provider must be one of {valid}, got {v}")
        return v.lower()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default="gpt-4o")
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    
    def is_configured(self) -> bool:
        return bool(self.api_key)


class MCPConfig(BaseModel):
    """MCP Server connection configuration."""
    
    host: str = Field(default_factory=lambda: os.getenv("MCP_HOST", "localhost"))
    port: int = Field(default_factory=lambda: _get_int_env("MCP_PORT", 5050))
    timeout: float = Field(default=30.0)
    retry_attempts: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class AgentConfig(BaseModel):
    """Main Chat Agent configuration."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    
    system_prompt: str = Field(
        default="""You are JARVIS, an intelligent AI assistant for desktop automation.

CRITICAL INSTRUCTIONS:
- ONLY respond with user-facing messages. NEVER output any internal reasoning, analysis, or thinking.
- When using tools, do NOT explain what you're doing or say "I will call...". Just silently call the tool and present the result.
- ONLY output the FINAL response to the user.
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
Examples:
- User says "list files in downloads" → Call resolve_path("downloads") → Use result in list_directory
- User says "organize my desktop" → Call resolve_path("desktop") → Use result in organize_folder

SPOTIFY LOGIN HANDLING:
When a user asks to log in or check Spotify status:
1. Call "checkSpotifyAuth"
2. If not authenticated, the tool returns a login_url - present it clearly with browser instructions
3. If authenticated, confirm their account status

RESPONSE FORMAT:
- Respond directly with what the user needs to know
- No meta-commentary or step-by-step explanations
- Example: User says "Check Spotify" → You respond with login URL or current status, nothing else"""
    )
    
    debug: bool = Field(default=False)
    log_transcripts: bool = Field(default=True)
    session_id: str = Field(default_factory=lambda: os.getenv("CHAT_SESSION_ID", "default"))
    context_cache_enabled: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_CACHE_ENABLED", "true").lower() == "true"
    )
    context_cache_max_turns: int = Field(default=20, ge=4)
    context_cache_summary_keep_last: int = Field(default=8, ge=2)
    context_token_budget: int = Field(default=3000, ge=256)
    context_dtype: str = Field(default_factory=lambda: os.getenv("CONTEXT_DTYPE", "fp16"))
    context_cache_path: str = Field(default_factory=lambda: os.getenv("CONTEXT_CACHE_PATH", ""))
    tool_retry_attempts: int = Field(default=2, ge=0)
    tool_retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    llm_response_cache_enabled: bool = Field(
        default_factory=lambda: _get_bool_env("LLM_RESPONSE_CACHE_ENABLED", True)
    )
    llm_response_cache_ttl_seconds: int = Field(
        default_factory=lambda: _get_int_env("LLM_RESPONSE_CACHE_TTL_SECONDS", 180),
        ge=1,
        le=3600,
    )
    llm_response_cache_max_entries: int = Field(
        default_factory=lambda: _get_int_env("LLM_RESPONSE_CACHE_MAX_ENTRIES", 256),
        ge=16,
        le=5000,
    )
    llm_response_cache_min_chars: int = Field(
        default_factory=lambda: _get_int_env("LLM_RESPONSE_CACHE_MIN_CHARS", 24),
        ge=1,
    )
    llm_response_cache_path: str = Field(
        default_factory=lambda: os.getenv("LLM_RESPONSE_CACHE_PATH", "")
    )
    llm_response_cache_allow_tool_providers: bool = Field(
        default_factory=lambda: _get_bool_env("LLM_RESPONSE_CACHE_ALLOW_TOOL_PROVIDERS", False)
    )

    @field_validator("context_dtype")
    @classmethod
    def validate_context_dtype(cls, v: str) -> str:
        normalized = v.lower().strip()
        if normalized not in SUPPORTED_CONTEXT_DTYPES:
            valid = sorted(SUPPORTED_CONTEXT_DTYPES)
            raise ValueError(f"context_dtype must be one of {valid}, got '{v}'")
        return normalized


def load_config(env_file: Optional[Path] = None) -> AgentConfig:
    """Load configuration from environment."""
    if env_file and env_file.exists():
        load_dotenv(env_file)
    
    return AgentConfig()
