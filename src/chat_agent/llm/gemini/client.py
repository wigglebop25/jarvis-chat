"""Gemini API client initialization and configuration."""

import logging
import warnings
from typing import Any

from .errors import GeminiConfigurationError

logger = logging.getLogger(__name__)


def setup_gemini_client(api_key: str, model: str = "gemini-1.5-flash"):
    """
    Initialize Gemini client and configuration.
    
    Args:
        api_key: Google API key for Gemini
        model: Model name to use
        
    Returns:
        Tuple of (GenerativeModel class, GenerationConfig class, initialized client)
        
    Raises:
        GeminiConfigurationError: If API key is not provided or invalid
    """
    if not api_key:
        raise GeminiConfigurationError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from google.generativeai.client import configure
        from google.generativeai.generative_models import GenerativeModel
        from google.generativeai.types import GenerationConfig
    
    try:
        configure(api_key=api_key)
    except Exception as e:
        raise GeminiConfigurationError(f"Failed to configure Gemini API: {e}") from e
    
    client = GenerativeModel(model)
    return GenerativeModel, GenerationConfig, client


def get_generation_config(
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Any:
    """
    Create a GenerationConfig for Gemini.
    
    Args:
        temperature: Temperature for sampling
        max_tokens: Maximum tokens in response
        
    Returns:
        GenerationConfig instance
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from google.generativeai.types import GenerationConfig
    
    return GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def get_available_models() -> list[str]:
    """
    Fetch available Gemini models.
    
    Returns:
        List of available model names, or empty list if fetch fails
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from google.generativeai.models import list_models
        
        models = list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                name = m.name.replace("models/", "")
                available.append(name)
        return available
    except Exception as e:
        logger.warning(f"Failed to fetch available models: {e}")
        return []


def get_available_models_detailed() -> list[dict[str, Any]]:
    """
    Fetch available models with metadata.
    
    Returns:
        List of model dicts with name, context limits, etc
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from google.generativeai.models import list_models
        
        models = list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append({
                    "name": m.name.replace("models/", ""),
                    "full_name": m.name,
                    "input_token_limit": getattr(m, "input_token_limit", 0),
                    "output_token_limit": getattr(m, "output_token_limit", 0),
                    "description": getattr(m, "description", ""),
                })
        return available
    except Exception as e:
        logger.warning(f"Failed to fetch model details: {e}")
        return []
