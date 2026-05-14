"""Tool definitions - System tools."""

SYSTEM_TOOLS = [
    {
        "name": "resolve_path",
        "description": "Resolve 'downloads', 'documents', 'desktop', 'home' or 'project' to full system paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["downloads", "documents", "desktop", "home", "project"],
                    "description": "Name: downloads, documents, desktop, home, or project",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_system_info",
        "description": "Get CPU, RAM, storage, and network status.",
        "parameters": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "description": "Sections: cpu, ram, storage, network.",
                    "items": {"type": "string"},
                }
            },
            "required": [],
        },
    },
    {
        "name": "control_volume",
        "description": "Manage system volume (get, set, up, down, mute, unmute).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "up", "down", "mute", "unmute"],
                },
                "level": {
                    "type": "integer",
                    "description": "Volume 0-100 (for 'set').",
                },
                "step": {
                    "type": "integer",
                    "description": "Step size (for 'up'/'down').",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "toggle_network",
        "description": "Enable/disable WiFi, Bluetooth, or Ethernet.",
        "parameters": {
            "type": "object",
            "properties": {
                "interface": {"type": "string", "enum": ["wifi", "bluetooth", "ethernet"]},
                "enable": {"type": "boolean"},
            },
            "required": ["interface", "enable"],
        },
    },
]
