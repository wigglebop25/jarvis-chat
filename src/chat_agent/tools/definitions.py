"""Provider-agnostic tool definitions."""


def get_tool_definitions() -> list[dict]:
    """
    Get list of available tool definitions.

    Returns:
        List of tool schema dictionaries compatible with multiple LLM providers.
    """
    return [
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
            "name": "control_spotify",
            "description": (
                "Control Spotify playback (play, pause, next, previous, current, search)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Playback action to execute.",
                        "enum": ["play", "pause", "next", "previous", "current", "search"],
                    },
                    "uri": {
                        "type": "string",
                        "description": "Optional Spotify URI to play.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query when action is search.",
                    },
                },
                "required": ["action"],
            },
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
