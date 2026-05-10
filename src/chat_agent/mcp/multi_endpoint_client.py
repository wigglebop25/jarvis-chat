"""
Multi-Endpoint MCP Client

Allows routing tool calls to different MCP servers based on tool category.
Supports multiple endpoints for different tool groups (system, spotify, etc).
"""

from typing import Any, Optional
import os

from .client import MCPClient


class MultiEndpointMCPClient:
    """Routes MCP calls to different endpoints based on tool name/category."""

    # Tool categories mapped to MCP server endpoints
    ENDPOINT_MAP = {
        # Spotify tools → Spotify MCP server (port 3000)
        "spotify": [
            "searchSpotify",
            "getNowPlaying",
            "getMyPlaylists",
            "getPlaylistTracks",
            "getRecentlyPlayed",
            "getUsersSavedTracks",
            "removeUsersSavedTracks",
            "getQueue",
            "getAvailableDevices",
            "playMusic",
            "pausePlayback",
            "skipToNext",
            "skipToPrevious",
            "createPlaylist",
            "addTracksToPlaylist",
            "resumePlayback",
            "addToQueue",
            "setVolume",
            "adjustVolume",
            "getAlbums",
            "getAlbumTracks",
            "saveOrRemoveAlbumForUser",
            "checkUsersSavedAlbums",
            "getPlaylist",
            "updatePlaylist",
            "removeTracksFromPlaylist",
            "reorderPlaylistItems",
        ],
        # System tools → Rust MCP server (port 5050)
        "system": [
            "get_system_info",
            "control_volume",
            "toggle_network",
            "list_directory",
            "organize_folder",
            "resolve_path",
            "control_bluetooth_device",
            "jarvis/route_and_call",
        ],
    }

    def __init__(
        self,
        system_endpoint: Optional[str] = None,
        spotify_endpoint: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize multi-endpoint MCP client.

        Args:
            system_endpoint: URL for system MCP server (default: env MCP_SERVER_URL)
            spotify_endpoint: URL for Spotify MCP server (default: http://127.0.0.1:3000)
            timeout: Request timeout in seconds
        """
        self.system_endpoint = system_endpoint or os.getenv(
            "MCP_SERVER_URL", "http://127.0.0.1:5050"
        )
        self.spotify_endpoint = spotify_endpoint or os.getenv(
            "SPOTIFY_MCP_URL", "http://127.0.0.1:3000"
        )
        self.timeout = timeout

        # Initialize clients for each endpoint
        self.system_client = MCPClient(base_url=self.system_endpoint, timeout=timeout)
        self.spotify_client = MCPClient(base_url=self.spotify_endpoint, timeout=timeout)

    def _get_endpoint_client(self, tool_name: str) -> MCPClient:
        """Determine which client to use based on tool name."""
        # Check if tool is in Spotify category
        if tool_name in self.ENDPOINT_MAP["spotify"]:
            return self.spotify_client
        # Default to system client
        return self.system_client

    async def call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Call an MCP method, routing to the appropriate endpoint.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout (uses client default if not specified)

        Returns:
            Result from the RPC call

        Raises:
            RuntimeError: If the RPC call fails
        """
        # Special handling for route_and_call - always use system endpoint
        if method == "jarvis/route_and_call":
            return await self.system_client.call(method, params, timeout)

        # For tools/list, return aggregated results from all endpoints
        if method == "tools/list":
            return await self._list_all_tools(timeout)

        # For other methods, route based on method name
        client = self.system_client
        if params and "name" in params:
            tool_name = params["name"]
            client = self._get_endpoint_client(tool_name)

        return await client.call(method, params, timeout)

    async def _list_all_tools(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """List tools from all available endpoints."""
        all_tools = {"tools": []}

        # Get tools from system endpoint
        try:
            system_result = await self.system_client.call("tools/list", {}, timeout)
            if isinstance(system_result, dict) and "tools" in system_result:
                all_tools["tools"].extend(system_result["tools"])
        except Exception as exc:
            # Log but don't fail - system endpoint might be unavailable
            print(f"Warning: Could not fetch system tools: {exc}")

        # Get tools from Spotify endpoint
        try:
            spotify_result = await self.spotify_client.call("tools/list", {}, timeout)
            if isinstance(spotify_result, dict) and "tools" in spotify_result:
                all_tools["tools"].extend(spotify_result["tools"])
        except Exception as exc:
            # Log but don't fail - Spotify endpoint might be unavailable
            print(f"Warning: Could not fetch Spotify tools: {exc}")

        return all_tools

    async def health_check(self) -> dict[str, bool]:
        """Check health of all endpoints."""
        health = {
            "system": await self.system_client.health_check(),
            "spotify": await self.spotify_client.health_check(),
        }
        return health

    async def close(self):
        """Close all HTTP clients."""
        await self.system_client.close()
        await self.spotify_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
