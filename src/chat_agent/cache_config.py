"""Cache configuration for LLM responses and context."""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _get_bool_env(name: str, default: bool) -> bool:
    """Get boolean environment variable with default fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _get_int_env(name: str, default: int) -> int:
    """Get integer environment variable with default fallback."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


class CacheConfig(BaseModel):
    """Configuration for caching system."""
    
    # LLM Response Cache
    llm_response_cache_enabled: bool = Field(
        default_factory=lambda: _get_bool_env("LLM_RESPONSE_CACHE_ENABLED", True)
    )
    llm_response_cache_ttl_seconds: int = Field(
        default_factory=lambda: _get_int_env("LLM_RESPONSE_CACHE_TTL_SECONDS", 180),
        ge=1,
        le=3600,
    )
    llm_response_cache_max_entries: int = Field(
        default_factory=lambda: _get_int_env("LLM_RESPONSE_CACHE_ENTRIES", 256),
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
    
    # Context Cache
    context_cache_enabled: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_CACHE_ENABLED", "true").lower() == "true"
    )
    context_cache_max_turns: int = Field(default=20, ge=4)
    context_cache_summary_keep_last: int = Field(default=8, ge=2)
    context_token_budget: int = Field(default=3000, ge=256)
    context_cache_path: str = Field(
        default_factory=lambda: os.getenv("CONTEXT_CACHE_PATH", "")
    )
    
    def __init__(self, **data):
        """Initialize cache config and validate directories."""
        super().__init__(**data)
        
        # Validate context cache directory if enabled
        if self.context_cache_enabled and self.context_cache_path:
            cache_dir = Path(self.context_cache_path).expanduser()
            _validate_cache_directory(cache_dir)
        
        # Validate LLM response cache directory if enabled
        if self.llm_response_cache_enabled and self.llm_response_cache_path:
            cache_dir = Path(self.llm_response_cache_path).expanduser()
            _validate_cache_directory(cache_dir)
