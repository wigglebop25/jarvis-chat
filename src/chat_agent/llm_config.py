"""LLM provider configuration."""

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


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
