"""
Chat Agent Configuration

Configuration management for the JARVIS Chat Agent.
Loads settings from environment variables and .env files.

This module serves as a facade that re-exports configuration classes
organized by domain:
- LLM Configuration: LLMConfig, OpenAIConfig
- MCP Configuration: MCPConfig  
- Cache Configuration: CacheConfig
- Agent Configuration: AgentConfig
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


def _validate_cache_directory(cache_dir: Path) -> None:
    """
    Validate that cache directory exists and is writable.
    
    Raises RuntimeError if directory cannot be created or written to.
    """
    try:
        # Create directory if needed
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Test writability by creating and deleting a test file
        test_file = cache_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"✓ Cache directory validated: {cache_dir}")
    except (OSError, PermissionError) as e:
        error_msg = (
            f"Cache directory {cache_dir} is not writable. "
            f"Check permissions or set JARVIS_CACHE_DIR to a writable path. "
            f"Error: {e}"
        )
        logger.error(f"✗ {error_msg}")
        raise RuntimeError(error_msg) from e


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


# Import and re-export specialized config classes
from .llm_config import LLMConfig, OpenAIConfig  # noqa: F401, E402
from .mcp_config import MCPConfig  # noqa: F401, E402
from .cache_config import CacheConfig  # noqa: F401, E402
from .agent_config import AgentConfig  # noqa: F401, E402


def load_config(env_file: Optional[Path] = None) -> AgentConfig:
    """
    Load configuration from environment.
    
    Args:
        env_file: Optional path to .env file to load
        
    Returns:
        Fully initialized AgentConfig instance
    """
    if env_file and env_file.exists():
        load_dotenv(env_file)

    return AgentConfig()


__all__ = [
    # Helpers
    "_validate_cache_directory",
    "_get_bool_env",
    "_get_int_env",
    # LLM Configuration
    "LLMConfig",
    "OpenAIConfig",
    # MCP Configuration
    "MCPConfig",
    # Cache Configuration
    "CacheConfig",
    # Agent Configuration
    "AgentConfig",
    # Main entry point
    "load_config",
]
