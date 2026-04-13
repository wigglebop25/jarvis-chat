from typing import Any

from .base import LLMProvider, LLMProviderError

# Lazy imports - only load what's needed
PROVIDER_CLASSES = {}


def _load_provider(name: str):
    """Lazy load provider class."""
    if name in PROVIDER_CLASSES:
        return PROVIDER_CLASSES[name]
    
    try:
        if name == "ollama":
            from .ollama import OllamaProvider
            PROVIDER_CLASSES["ollama"] = OllamaProvider
        elif name == "openai":
            from .openai import OpenAIProvider
            PROVIDER_CLASSES["openai"] = OpenAIProvider
        elif name == "gemini":
            from .gemini import GeminiProvider
            PROVIDER_CLASSES["gemini"] = GeminiProvider
        elif name == "copilot":
            from .copilot import CopilotProvider
            PROVIDER_CLASSES["copilot"] = CopilotProvider
        else:
            raise LLMProviderError(f"Unknown provider: {name}")
        
        return PROVIDER_CLASSES[name]
    except ImportError as e:
        raise LLMProviderError(
            f"Provider '{name}' not available. Install required dependencies:\n"
            f"  ollama: no extra deps needed\n"
            f"  openai: pip install openai\n"
            f"  gemini: pip install google-generativeai\n"
            f"  copilot: Copilot CLI (https://github.com/github/copilot-cli)\n"
            f"Error: {e}"
        ) from e


def create_provider(name: str, **kwargs) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        name: Provider name ('ollama', 'openai', 'gemini', 'copilot')
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLMProvider instance

    Raises:
        LLMProviderError: If provider name is unknown or creation fails
    """
    try:
        provider_class = _load_provider(name)
        return provider_class(**kwargs)
    except TypeError as e:
        raise LLMProviderError(f"Failed to instantiate {name} provider: {e}") from e


def get_available_providers() -> list[str]:
    """Get list of available provider names."""
    return ["ollama", "openai", "gemini", "copilot"]
