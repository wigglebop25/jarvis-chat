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


load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM provider selection and parameters."""
    
    provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama', 'openai', 'gemini', 'copilot'"
    )
    model: str = Field(
        default="llama3",
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
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
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
If a task requires using a tool, indicate which tool should be used.

Be concise, helpful, and friendly. Acknowledge commands and confirm actions."""
    )
    
    debug: bool = Field(default=False)
    log_transcripts: bool = Field(default=True)


def load_config(env_file: Optional[Path] = None) -> AgentConfig:
    """Load configuration from environment."""
    if env_file and env_file.exists():
        load_dotenv(env_file)
    
    return AgentConfig()
