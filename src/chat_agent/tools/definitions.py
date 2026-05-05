"""Provider-agnostic tool definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_DEFAULT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "resolve_path",
        "description": "ALWAYS use this first when user says 'downloads', 'documents', 'desktop' or 'home'. Resolves user-friendly names to full system paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["downloads", "documents", "desktop", "home", "project"],
                    "description": "User-friendly name: downloads, documents, desktop, home, or project",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_system_info",
        "description": (
            "Get system information including CPU usage, RAM, storage, and network status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "description": "Optional sections: cpu, ram, storage, network.",
                    "items": {"type": "string"},
                }
            },
            "required": [],
        },
    },
    {
        "name": "control_volume",
        "description": "Control system volume (get, set, up, down, mute, unmute).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform.",
                    "enum": ["get", "set", "up", "down", "mute", "unmute"],
                },
                "level": {
                    "type": "integer",
                    "description": "Volume level (0-100) when action is set.",
                },
                "step": {
                    "type": "integer",
                    "description": "Step size for up/down actions.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "checkSpotifyAuth",
        "description": "Check if the user is authenticated with Spotify. Returns a login URL if not.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "authorizeSpotify",
        "description": "Initiate the Spotify authorization flow by opening a browser.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "logoutSpotify",
        "description": "Logs the user out of Spotify by clearing the authentication cache.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "searchSpotify",
        "description": "Search for tracks, albums, artists, or playlists on Spotify",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query", "type"]
        }
    },
    {
        "name": "getNowPlaying",
        "description": "Get information about the currently playing track on Spotify, including device and volume info",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "playMusic",
        "description": "Start playing a track, album, artist, or playlist on Spotify",
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {"type": "string"},
                "type": {"type": "string"},
                "id": {"type": "string"},
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "pausePlayback",
        "description": "Pause the currently playing track on Spotify",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "skipToNext",
        "description": "Skip to the next track in the current playback queue",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "skipToPrevious",
        "description": "Skip to the previous track in the current playback queue",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "setVolume",
        "description": "Set the playback volume to a specific percentage (requires Spotify Premium)",
        "parameters": {
            "type": "object",
            "properties": {
                "volumePercent": {"type": "integer"},
                "deviceId": {"type": "string"}
            },
            "required": ["volumePercent"]
        }
    },
    {
        "name": "getAvailableDevices",
        "description": "Get information about the user's available Spotify Connect devices",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "toggle_network",
        "description": "Enable or disable WiFi, Bluetooth, or Ethernet.",
        "parameters": {
            "type": "object",
            "properties": {
                "interface": {
                    "type": "string",
                    "description": "Network interface to control.",
                    "enum": ["wifi", "bluetooth", "ethernet"],
                },
                "enable": {
                    "type": "boolean",
                    "description": "True to enable, false to disable.",
                },
            },
            "required": ["interface", "enable"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and folders in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "FULL directory path.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and folders.",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum number of entries to return.",
                },
                "directories_only": {
                    "type": "boolean",
                    "description": "Return directories only.",
                },
                "files_only": {
                    "type": "boolean",
                    "description": "Return files only.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "organize_folder",
        "description": (
            "Organize files in a folder by extension, type, or date "
            "within allowlisted directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Folder path to organize.",
                },
                "strategy": {
                    "type": "string",
                    "description": "Organization strategy.",
                    "enum": ["extension", "type", "date"],
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Include nested files.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview only without moving files.",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files.",
                },
            },
            "required": ["path"],
        },
    },
]


def get_tool_definitions() -> list[dict[str, Any]]:
    """
    Get list of available tool definitions.

    Returns:
        List of tool schema dictionaries compatible with multiple LLM providers.
    """
    return deepcopy(_DEFAULT_TOOL_DEFINITIONS)


def normalize_mcp_tool_definitions(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize MCP `tools/list` payloads into provider-agnostic tool definitions.

    Supports both:
    - HTTP compatibility shape: {"type":"function","function":{...}}
    - MCP stdio shape: {"name":"...","description":"...","inputSchema":{...}}
    """
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue

        name: Any = None
        description: Any = ""
        parameters: Any = {}

        function_payload = raw_tool.get("function")
        if isinstance(function_payload, dict):
            name = function_payload.get("name")
            description = function_payload.get("description", "")
            parameters = function_payload.get("parameters", {})
        else:
            name = raw_tool.get("name")
            description = raw_tool.get("description", "")
            parameters = raw_tool.get("parameters")
            if parameters is None:
                parameters = raw_tool.get("inputSchema", {})

        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        if not clean_name or clean_name in seen_names:
            continue

        if not isinstance(description, str):
            description = str(description)
        if not isinstance(parameters, dict):
            parameters = {}

        normalized.append(
            {
                "name": clean_name,
                "description": description,
                "parameters": parameters,
            }
        )
        seen_names.add(clean_name)

    return normalized
