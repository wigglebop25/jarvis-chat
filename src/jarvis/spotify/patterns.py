"""
src/jarvis/spotify/patterns.py
────────────────────────────────
Pre-compiled regex patterns for the Spotify fast-path.

All patterns are compiled once at import time (O(1) matching cost).
To add a new pattern: add it here and import it in handlers.py.
"""
import re
from typing import Optional

# ── Volume control ────────────────────────────────────────────────────────────

VOLUME_SET_RE: re.Pattern[str] = re.compile(
    r"(?:adjust|set)\s+(?:spotify\s+)?volume(?:\s+to)?\s+(\d{1,3})",
    re.IGNORECASE,
)
VOLUME_UP_RE: re.Pattern[str] = re.compile(
    r"\b(?:volume up|increase volume|turn up(?:\s+the)?\s+volume|louder)\b",
    re.IGNORECASE,
)
VOLUME_DOWN_RE: re.Pattern[str] = re.compile(
    r"\b(?:volume down|decrease volume|turn down(?:\s+the)?\s+volume|quieter)\b",
    re.IGNORECASE,
)

# ── Playback ──────────────────────────────────────────────────────────────────

PLAY_PLAYLIST_RE: re.Pattern[str] = re.compile(
    r"^(?:play|start)\s+(?:music\s+)?(?:in\s+(?:the\s+)?)?playlist\s+(.+)$"
    r"|^(?:play|start)\s+music\s+in\s+(?:the\s+)?(.+)$",
    re.IGNORECASE,
)
PLAY_TRACK_RE: re.Pattern[str] = re.compile(
    r"^(?:play|start)\s+(?:the\s+)?(.+)$",
    re.IGNORECASE,
)

# Track query values too generic to attempt a search
PLAY_TRACK_GENERIC: frozenset[str] = frozenset({"music", "song", "something", "anything"})

# ── getNowPlaying output field parsers ────────────────────────────────────────

NP_TRACK_ID_RE: re.Pattern[str] = re.compile(
    r"\*\*ID\*\*:\s*([A-Za-z0-9]+)"
)
NP_SHUFFLE_RE: re.Pattern[str] = re.compile(
    r"\*\*Shuffle\*\*:\s*(On|Off)", re.IGNORECASE
)
NP_TRACK_NAME_RE: re.Pattern[str] = re.compile(
    r'\*\*Track\*\*:\s*"(.+?)"'
)
NP_ARTIST_RE: re.Pattern[str] = re.compile(
    r"\*\*Artist\*\*:\s*(.+)"
)
NP_DURATION_RE: re.Pattern[str] = re.compile(
    r"\*\*Progress\*\*:\s*[\d:]+\s*/\s*([\d:]+)"
)

# ── Playlist tracks parser ────────────────────────────────────────────────────
# Matches lines like: 1. "Track Name" by Artist (3:45) - ID: trackId

PLAYLIST_TRACK_RE: re.Pattern[str] = re.compile(
    r'^\d+\.\s+"(.+?)"\s+by\s+(.+?)\s+\(([\d:]+)\)\s+-\s+ID:\s*(\S+)$'
)

# ── Active playlist cache ─────────────────────────────────────────────────────
# Set by handle_play_playlist() when a playlist starts via JARVIS.
# Used by handle_show_queue() to show accurate track order.
# Falls back to getQueue API if None (playlist started externally).

active_playlist: dict[str, Optional[str]] = {"id": None, "name": None}
