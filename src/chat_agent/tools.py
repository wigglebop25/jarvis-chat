"""
Tool Definitions

OpenAI-compatible tool definitions for MCP Server tools.
These definitions tell the LLM what tools are available and how to call them.
"""

from typing import Any

AVAILABLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get system information including CPU usage, RAM usage, storage space, and network status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["cpu", "ram", "storage", "network", "all"]
                        },
                        "description": "What information to include. Defaults to 'all'.",
                        "default": ["all"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_volume",
            "description": "Control system audio volume. Can set volume level, mute/unmute, or adjust up/down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "up", "down", "mute", "unmute", "get"],
                        "description": "The volume action to perform."
                    },
                    "level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Volume level (0-100). Only used with 'set' action."
                    },
                    "step": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Amount to adjust for up/down actions."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_spotify",
            "description": "Control Spotify playback (play, pause, next, previous, current, search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "next", "previous", "current", "search"],
                        "description": "The playback action to perform."
                    },
                    "uri": {
                        "type": "string",
                        "description": "Optional Spotify URI to play (track, album, or playlist)."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query when action is 'search'."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_network",
            "description": "Enable or disable WiFi, Bluetooth, or Ethernet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interface": {
                        "type": "string",
                        "enum": ["wifi", "bluetooth", "ethernet"],
                        "description": "Which network interface to control."
                    },
                    "enable": {
                        "type": "boolean",
                        "description": "True to enable, false to disable."
                    }
                },
                "required": ["interface", "enable"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "organize_folder",
            "description": "Organize files in a folder by extension, type, or date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Folder path to organize."
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["extension", "type", "date"],
                        "default": "extension",
                        "description": "Organization strategy."
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include nested files."
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": True,
                        "description": "Preview only without moving files."
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include hidden files."
                    }
                },
                "required": ["path"]
            }
        }
    }
]


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get all available tool definitions for OpenAI API."""
    return AVAILABLE_TOOLS


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Get a specific tool definition by name."""
    for tool in AVAILABLE_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def format_tool_result_for_display(tool_name: str, result: Any) -> str:
    """Format a tool result for user-friendly display."""
    if tool_name == "get_system_info" and isinstance(result, dict):
        lines = ["System Information:"]
        if "cpu" in result:
            lines.append(f"  CPU: {result['cpu']}% usage")
        if "ram" in result:
            ram = result["ram"]
            lines.append(f"  RAM: {ram.get('used_gb', 'N/A')} GB / {ram.get('total_gb', 'N/A')} GB ({ram.get('percent', 'N/A')}%)")
        if "storage" in result:
            for drive in result.get("storage", []):
                lines.append(f"  {drive.get('mount', 'Drive')}: {drive.get('free_gb', 'N/A')} GB free")
        return "\n".join(lines)
    
    if tool_name == "control_volume" and isinstance(result, dict):
        if "level" in result:
            return f"Volume set to {result['level']}%"
        if "muted" in result:
            return "Audio muted" if result["muted"] else "Audio unmuted"
        return str(result)
    
    if tool_name == "control_spotify" and isinstance(result, dict):
        if "track" in result:
            track = result["track"]
            return f"Now playing: {track.get('name', 'Unknown')} by {track.get('artist', 'Unknown')}"
        if "action" in result:
            return f"Spotify: {result['action']}"
        return str(result)

    if tool_name == "organize_folder" and isinstance(result, dict):
        if "error" in result:
            return f"Folder organizer error: {result['error']}"
        if result.get("dry_run", True):
            return f"Folder organizer preview complete. Planned moves: {result.get('planned_moves', 0)}."
        return f"Folder organization complete. Files moved: {result.get('moved_files', 0)}."
    
    return str(result)
