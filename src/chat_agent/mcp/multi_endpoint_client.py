"""
Multi-Endpoint MCP Client

Allows routing tool calls to different MCP servers based on tool category.
Supports multiple endpoints for different tool groups (system, spotify, etc).
"""

import logging
from typing import Any, Optional
import os
import json
from pathlib import Path

from .client import MCPClient
from .stdio_client import StdioMCPClient

logger = logging.getLogger(__name__)

SPOTIFY_TOOL_NAMES = {
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
    "resumePlayback",
    "skipToNext",
    "skipToPrevious",
    "createPlaylist",
    "addTracksToPlaylist",
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
}


def _workspace_root() -> Path:
    """Resolve the workspace root that contains jarvis-chat, jarvis-skills, and spotify-mcp-server."""
    return Path(__file__).resolve().parents[4]


def _resolve_path(value: str | Path | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())

class MultiEndpointMCPClient:
    """Routes MCP calls to different endpoints based on tool name/category."""

    def __init__(
        self,
        system_endpoint: Optional[str] = None,
        spotify_endpoint: Optional[str] = None,
        system_transport: str = "http",
        spotify_transport: str = "http",
        timeout: int = 30,
    ):
        """
        Initialize multi-endpoint MCP client.
        """
        self.system_endpoint: str = system_endpoint or os.getenv("MCP_SYSTEM_URL") or "http://127.0.0.1:5050"
        self.spotify_endpoint: str = spotify_endpoint or os.getenv("MCP_SPOTIFY_URL") or "http://127.0.0.1:3000"
        self.timeout = timeout

        # Initialize clients based on transport
        self.system_client = self._create_client(self.system_endpoint, system_transport, "SYSTEM")
        self.spotify_client = self._create_client(self.spotify_endpoint, spotify_transport, "SPOTIFY")
        
        # Tool name to client mapping (populated during tools/list)
        self._tool_registry: dict[str, Any] = {}

    def _create_client(self, endpoint: str, transport: str, prefix: str) -> Any:
        transport = transport.lower()
        if transport == "stdio":
            # Priority: explicit env vars -> MCP_CONFIG_PATH or jarvis-skills/mcp.json -> defaults
            env_cmd = os.getenv(f"{prefix}_MCP_COMMAND")
            env_args = os.getenv(f"{prefix}_MCP_ARGS")
            env_cwd = os.getenv(f"{prefix}_MCP_CWD")
            command: str
            args: list[str]
            cwd: str | Path | None
            extra_env: dict[str, str] | None
            if env_cmd:
                command = env_cmd.strip()
                args = [arg for arg in (env_args.split() if env_args else []) if arg]
                cwd = env_cwd
                extra_env = None
            else:
                # Try MCP_CONFIG_PATH then common repo locations
                mcp_config = os.getenv("MCP_CONFIG_PATH")
                candidates = [mcp_config] if mcp_config else []
                candidates += ["mcp.json", "../jarvis-skills/mcp.json", "jarvis-skills/mcp.json"]
                found = None
                for c in candidates:
                    if not c:
                        continue
                    p = Path(c)
                    if p.exists():
                        found = p
                        break
                if found:
                    try:
                        data = json.loads(found.read_text(encoding="utf-8"))
                        servers = data.get("mcpServers", {})
                        key = "spotify" if prefix == "SPOTIFY" else "jarvis-skills"
                        entry = servers.get(key, {})
                        command = str(entry.get("command") or ("node" if prefix == "SPOTIFY" else "cargo"))
                        args = [str(arg) for arg in (entry.get("args", []) or []) if arg is not None]
                        cwd = env_cwd or entry.get("cwd")
                        raw_env = entry.get("env") or None
                        extra_env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else None
                    except Exception:
                        command = os.getenv(f"{prefix}_MCP_COMMAND", "node" if prefix == "SPOTIFY" else "cargo").strip()
                        args = [arg for arg in (os.getenv(f"{prefix}_MCP_ARGS", "").split()) if arg]
                        cwd = env_cwd
                        extra_env = None
                else:
                    command = os.getenv(f"{prefix}_MCP_COMMAND", "node" if prefix == "SPOTIFY" else "cargo").strip()
                    args = [arg for arg in (os.getenv(f"{prefix}_MCP_ARGS", "").split()) if arg]
                    cwd = env_cwd
                    extra_env = None

            base_dir = _workspace_root()
            resolved_cwd = _resolve_path(cwd, base_dir) if cwd else None
            resolved_command = _resolve_path(command, base_dir) if (os.sep in command or "/" in command or "\\" in command) else command
            if resolved_command is None:
                resolved_command = command

            resolved_args: list[str] = []
            for arg in args:
                resolved_arg = _resolve_path(arg, base_dir) if (os.sep in arg or "/" in arg or "\\" in arg) else arg
                if resolved_arg is None:
                    resolved_arg = arg
                resolved_args.append(resolved_arg)
            logger.info(
                "Configured %s MCP stdio client: command=%s cwd=%s",
                prefix.lower(),
                resolved_command,
                resolved_cwd or "<inherit>",
            )

            return StdioMCPClient(
                command=resolved_command,
                args=resolved_args,
                timeout=self.timeout,
                cwd=resolved_cwd,
                env=extra_env,
            )
        else:
            return MCPClient(base_url=endpoint, timeout=self.timeout)

    async def _list_all_tools(self, timeout: Optional[int] = None) -> dict[str, Any]:
        """List tools from all available endpoints and update registry."""
        all_tools: dict[str, list[Any]] = {"tools": []}
        new_registry: dict[str, Any] = {}

        # Helper to fetch and register
        async def fetch_from(client, name):
            try:
                logger.info(f"Fetching tools from {name} endpoint...")
                result = await client.call("tools/list", {}, timeout)
                if isinstance(result, dict) and "tools" in result:
                    tools = result["tools"]
                    logger.info(f"Found {len(tools)} tools from {name}")
                    for tool in tools:
                        tool_name = tool.get("name")
                        if tool_name:
                            new_registry[tool_name] = client
                    return tools
            except Exception as exc:
                logger.warning(f"Could not fetch {name} tools: {exc}")
            return []

        # Gather tools from both endpoints
        system_tools = await fetch_from(self.system_client, "system")
        spotify_tools = await fetch_from(self.spotify_client, "spotify")
        
        all_tools["tools"].extend(system_tools)
        all_tools["tools"].extend(spotify_tools)

        # Atomic update of the registry
        self._tool_registry = new_registry
        logger.info(f"Total tools registered: {len(self._tool_registry)}")

        return all_tools

    async def call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Call an MCP method, routing to the appropriate endpoint."""
        # Special handling for route_and_call - always use system endpoint
        if method == "jarvis/route_and_call":
            return await self.system_client.call(method, params, timeout)

        if method == "tools/list":
            return await self._list_all_tools(timeout)

        # For tools/call, route based on registry
        if method == "tools/call" and params and "name" in params:
            tool_name = params["name"]
            client = self._tool_registry.get(tool_name)
            if client:
                return await client.call(method, params, timeout)
            
            if tool_name in SPOTIFY_TOOL_NAMES:
                logger.info(f"Tool {tool_name} not in registry, using spotify client fallback")
                return await self.spotify_client.call(method, params, timeout)

            # Fallback to system if not registered
            logger.warning(f"Tool {tool_name} not in registry, falling back to system client")
            return await self.system_client.call(method, params, timeout)

        # Default fallback
        return await self.system_client.call(method, params, timeout)

    async def health_check(self) -> dict[str, bool]:
        """Check health of all endpoints."""
        return {
            "system": await self.system_client.health_check(),
            "spotify": await self.spotify_client.health_check(),
        }

    async def close(self):
        """Close all clients."""
        await self.system_client.close()
        await self.spotify_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
