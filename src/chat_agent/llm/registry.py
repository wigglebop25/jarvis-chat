from typing import Any

from .base import LLMProvider, LLMProviderError

PROVIDER_MODULES = {
    "ollama": "chat_agent.llm.ollama",
    "openai": "chat_agent.llm.openai",
    "gemini": "chat_agent.llm.gemini",
    "copilot": "chat_agent.llm.copilot",
}


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
    if name not in PROVIDER_MODULES:
        raise LLMProviderError(
            f"Unknown provider: {name}. Available providers: {', '.join(PROVIDER_MODULES.keys())}"
        )

    module_path = PROVIDER_MODULES[name]

    try:
        parts = module_path.rsplit(".", 1)
        module_name = parts[-1]
        import_path = parts[0]

        module = __import__(import_path, fromlist=[module_name])

        class_name = "".join(word.capitalize() for word in module_name.split("_")) + "Provider"
        provider_class = getattr(module, class_name, None)

        if not provider_class:
            raise LLMProviderError(f"Provider class {class_name} not found in {import_path}")

        return provider_class(**kwargs)

    except ImportError as e:
        raise LLMProviderError(f"Failed to import {name} provider: {e}") from e
    except TypeError as e:
        raise LLMProviderError(f"Failed to instantiate {name} provider: {e}") from e


def get_available_providers() -> list[str]:
    """Get list of available provider names."""
    return list(PROVIDER_MODULES.keys())
