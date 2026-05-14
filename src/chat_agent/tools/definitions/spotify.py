"""Tool definitions - Spotify tools."""

SPOTIFY_TOOLS = [
    {
        "name": "checkSpotifyAuth",
        "description": "Check Spotify auth status. Returns login URL if needed.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "authorizeSpotify",
        "description": "Start Spotify login flow.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "logoutSpotify",
        "description": "Clear Spotify auth cache.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "searchSpotify",
        "description": "Search Spotify for tracks, albums, artists, or playlists.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string", "description": "track, album, artist, or playlist"},
                "limit": {"type": "integer"}
            },
            "required": ["query", "type"]
        }
    },
    {
        "name": "getNowPlaying",
        "description": "Get current Spotify track and device info.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "getQueue",
        "description": "Get the currently playing track and the next items in the queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of items to show (1-50)"}
            }
        }
    },
    {
        "name": "getMyPlaylists",
        "description": "Get a list of the current user's playlists.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of playlists (1-50)"}
            }
        }
    },
    {
        "name": "getPlaylistTracks",
        "description": "Get tracks in a Spotify playlist.",
        "parameters": {
            "type": "object",
            "properties": {
                "playlistId": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"}
            },
            "required": ["playlistId"]
        }
    },
    {
        "name": "getRecentlyPlayed",
        "description": "Get recently played tracks.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"}
            }
        }
    },
    {
        "name": "getUsersSavedTracks",
        "description": "Get user's liked songs.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "offset": {"type": "integer"}
            }
        }
    },
    {
        "name": "playMusic",
        "description": "Play Spotify track, album, artist, or playlist by URI or ID.",
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
        "description": "Pause Spotify playback.",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "skipToNext",
        "description": "Skip to next Spotify track.",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "skipToPrevious",
        "description": "Skip to previous Spotify track.",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceId": {"type": "string"}
            }
        }
    },
    {
        "name": "setVolume",
        "description": "Set Spotify volume (0-100).",
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
        "description": "List available Spotify Connect devices.",
        "parameters": {"type": "object", "properties": {}}
    },
]
