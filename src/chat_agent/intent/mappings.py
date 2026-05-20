"""
Intent Mappings

Logic for mapping intents and parameters to specific MCP tools.
"""

import re
import os
from pathlib import Path
from typing import Optional

from ..models import Intent, IntentType


def get_tool_name_for_intent(intent: Intent) -> Optional[str]:
    """Map an intent to its corresponding MCP tool name."""
    text = intent.raw_text.lower()
    
    # 1. Spotify / Music Specific Logic (High Priority Phrasings)
    if "spotify" in text or intent.type in [IntentType.MUSIC_CONTROL, IntentType.SPOTIFY_INFO]:
        # Special Collections
        if "liked songs" in text:
            if any(w in text for w in ["track", "song", "content", "inside", "show", "list"]):
                return "getUsersSavedTracks"
        
        # Playback Navigation
        if "previous" in text or "back" in text: return "skipToPrevious"
        if "next" in text or "skip" in text: return "skipToNext"
        if "pause" in text or "stop" in text: return "pausePlayback"
        if "resume" in text: return "resumePlayback"
        
        # Volume (Spotify-specific)
        if "volume" in text:
            if any(w in text for w in ["increase", "decrease", "up", "down", "by"]):
                return "adjustVolume"
            if any(w in text for w in ["set", "to", "percent", "%"]):
                return "setVolume"
        
        # Info
        if any(w in text for w in ["queue", "upcoming"]): return "getQueue"
        if "playlist" in text:
            if any(w in text for w in ["track", "song", "content", "inside"]): return "getPlaylistTracks"
            return "getMyPlaylists"
        if any(w in text for w in ["playing", "listening", "current track", "current song"]): return "getNowPlaying"
        if "device" in text: return "getAvailableDevices"
        if "search" in text or "find" in text: return "searchSpotify"
        
        if intent.type == IntentType.MUSIC_CONTROL: return "playMusic"

    # 2. General Tool Mapping
    tool_mapping = {
        IntentType.SYSTEM_INFO: "get_system_info",
        IntentType.VOLUME_CONTROL: "control_volume",
        IntentType.NETWORK_TOGGLE: "toggle_network",
        IntentType.DIRECTORY_LIST: "list_directory",
        IntentType.FILE_ORGANIZATION: "organize_folder",
        IntentType.PATH_RESOLVE: "resolve_path",
        IntentType.BLUETOOTH_CONTROL: "control_bluetooth_device",
    }
    
    # Special case: "current system volume"
    if intent.type == IntentType.VOLUME_CONTROL or "volume" in text:
        if "system" in text or "current" in text: return "control_volume"
        
    return tool_mapping.get(intent.type)


def map_intent_params_to_tool(intent: Intent) -> dict:
    """
    Map intent parameters to tool-expected parameters.
    """
    params = dict(intent.parameters)
    text = intent.raw_text.lower()
    tool_name = get_tool_name_for_intent(intent)
    
    if not tool_name:
        return params

    if tool_name == "control_volume":
        if "direction" in params:
            params["action"] = params.pop("direction")
        elif "level" in params:
            params["action"] = "set"
        elif "current" in text or "what" in text:
            params["action"] = "get"
        else:
            params["action"] = "get"
            
        if "level" in params:
            try: params["level"] = int(params["level"])
            except: params.pop("level", None)

    elif tool_name == "setVolume":
        match = re.search(r"(\d+)", text)
        if match: params["volumePercent"] = int(match.group(1))

    elif tool_name == "adjustVolume":
        match = re.search(r"(\d+)", text)
        if match:
            val = int(match.group(1))
            if any(w in text for w in ["decrease", "down", "lower", "reduce"]): val = -val
            params["adjustment"] = val

    elif tool_name == "getPlaylistTracks":
        match = re.search(r"playlist\s+[\"']?(.*?)[\"']?$", text)
        if match: params["playlistId"] = match.group(1).strip()

    elif tool_name == "playMusic":
        if "playlist" in text:
            match = re.search(r"playlist\s+[\"']?(.*?)[\"']?$", text)
            if match:
                params["uri"] = match.group(1).strip()
                params["type"] = "playlist"
        else:
            params["type"] = "track"

    elif tool_name == "searchSpotify":
        match = re.search(r"(?:search|find)\s+(?:for\s+)?(?:the\s+)?(.*?)\s+on\s+spotify", text)
        if match:
            query = match.group(1).strip()
            params["query"] = query
            if "album" in text: params["type"] = "album"
            elif "artist" in text: params["type"] = "artist"
            else: params["type"] = "track"

    elif intent.type == IntentType.NETWORK_TOGGLE:
        if "device" in params: params["interface"] = params.pop("device")
        if "state" in params: params["enable"] = params.pop("state") == "on"
            
    elif tool_name == "get_system_info":
        include = []
        if "cpu" in text or "processor" in text: include.append("cpu")
        if "ram" in text or "memory" in text: include.append("ram")
        if "storage" in text or "disk" in text or "drive" in text: include.append("storage")
        if "network" in text or "wifi" in text: include.append("network")
        if not include: include = ["cpu", "ram", "storage", "network"]
        params["include"] = include
            
    elif tool_name == "list_directory":
        def _to_drive(letter: str) -> str:
            return f"{letter.upper()}:\\"

        path = None
        for p in [r"\b([a-z])\s*:", r"drive\s*([a-z])\b"]:
            m = re.search(p, text)
            if m: path = _to_drive(m.group(1)); break
            
        if not path and "downloads" in text:
            path = str(Path.home() / "Downloads")

        params["path"] = path or str(Path.home())
        params["include_hidden"] = params.get("include_hidden", "false") == "true"
        params["max_entries"] = 200
        
    elif tool_name == "resolve_path":
        for folder in ["downloads", "documents", "desktop", "home", "project"]:
            if folder in text:
                params["name"] = folder
                break
                
    elif tool_name == "control_bluetooth_device":
        if any(w in text for w in ["list", "show", "see", "find"]):
            params["action"] = "list"
    
    return params
