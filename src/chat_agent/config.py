"""
Chat Agent Configuration

Configuration management for the JARVIS Chat Agent.
Loads settings from environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from .context_dtype import SUPPORTED_CONTEXT_DTYPES


load_dotenv()


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
    
    def get_provider_kwargs(self) -> dict:
        """Get provider-specific kwargs for create_provider()."""
        kwargs = {"model": self.model}
        
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
    
    host: str = Field(default="localhost")
    port: int = Field(default=5050)
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

You help users control their computer through voice commands. You can:
- Get system information (CPU, RAM, storage, network)
- Control system volume (up, down, mute, set level)
- Control music playback (Spotify play/pause, skip, etc.)
- Toggle WiFi and Bluetooth
- Provide helpful information and answers

When the user asks you to do something, determine the appropriate action and respond naturally.
If a task requires a tool, call the tool directly instead of describing which tool to use.
Never output internal reasoning or tool-selection analysis to the user.

Be concise, helpful, and friendly. Acknowledge commands and confirm actions."""
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
