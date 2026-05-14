"""MCP Server connection configuration."""

import os

from pydantic import BaseModel, Field


def _get_int_env(name: str, default: int) -> int:
    """Get integer environment variable with default fallback."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class MCPConfig(BaseModel):
    """MCP Server connection configuration."""

    host: str = Field(default_factory=lambda: os.getenv("MCP_HOST", "localhost"))
    port: int = Field(default_factory=lambda: _get_int_env("MCP_PORT", 5050))
    timeout: float = Field(default=30.0)
    retry_attempts: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    # Multi-endpoint support
    system_host: str = Field(default_factory=lambda: os.getenv("MCP_SYSTEM_HOST", "localhost"))
    system_port: int = Field(default_factory=lambda: _get_int_env("MCP_SYSTEM_PORT", 5050))
    system_transport: str = Field(default_factory=lambda: os.getenv("MCP_SYSTEM_TRANSPORT", "http"))
    
    spotify_host: str = Field(default_factory=lambda: os.getenv("MCP_SPOTIFY_HOST", "localhost"))
    spotify_port: int = Field(default_factory=lambda: _get_int_env("MCP_SPOTIFY_PORT", 3000))
    spotify_transport: str = Field(default_factory=lambda: os.getenv("MCP_SPOTIFY_TRANSPORT", "http"))
    
    # Enable multi-endpoint routing
    multi_endpoint_enabled: bool = Field(
        default_factory=lambda: os.getenv("MCP_MULTI_ENDPOINT_ENABLED", "true").lower() == "true"
    )
    
    @property
    def url(self) -> str:
        """Primary MCP server URL (system)."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def system_url(self) -> str:
        """System MCP server URL."""
        return f"http://{self.system_host}:{self.system_port}"
    
    @property
    def spotify_url(self) -> str:
        """Spotify MCP server URL."""
        return f"http://{self.spotify_host}:{self.spotify_port}"
