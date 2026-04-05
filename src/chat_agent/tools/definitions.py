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
            "description": "Get current system information including CPU, memory, and disk usage",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "control_volume",
            "description": "Control system volume (get, set, mute, unmute)",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'get', 'set', 'mute', 'unmute'",
                        "enum": ["get", "set", "mute", "unmute"],
                    },
                    "level": {
                        "type": "number",
                        "description": "Volume level (0-100) for 'set' action",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "toggle_network",
            "description": "Get network status or toggle WiFi/Bluetooth",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'status', 'wifi_toggle', 'bluetooth_toggle'",
                        "enum": ["status", "wifi_toggle", "bluetooth_toggle"],
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "num_results": {
                        "type": "number",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "play_music",
            "description": "Play music on Spotify or system player",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song, artist, or playlist to play",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Music provider",
                        "enum": ["spotify", "system"],
                        "default": "spotify",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates",
                    },
                },
                "required": ["location"],
            },
        },
    ]
