"""Gemini-specific exceptions."""

from ..base import LLMProviderError, LLMConfigurationError


class GeminiError(LLMProviderError):
    """Base exception for Gemini-specific errors."""


class GeminiConfigurationError(LLMConfigurationError):
    """Raised when Gemini provider is not properly configured."""


class GeminiConnectionError(GeminiError):
    """Raised when connection to Gemini API fails."""


class GeminiTimeoutError(GeminiError):
    """Raised when Gemini API request times out."""


class GeminiRateLimitError(GeminiError):
    """Raised when rate limited by Gemini API."""


class GeminiAPIError(GeminiError):
    """Raised when Gemini API returns an error."""
