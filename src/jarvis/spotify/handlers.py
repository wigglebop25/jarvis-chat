"""
src/jarvis/spotify/handlers.py
───────────────────────────────
Single-responsibility handlers for each Spotify fast-path command.

Each function:
  - Accepts only what it needs (router, transcript/normalized string)
  - Returns a formatted string response, or None to signal no match
  - Has zero side-effects except updating active_playlist cache
"""

import re
from typing import Any, Optional

from chat_agent.spotify_helpers import resolve_playlist_match
from chat_agent.tools.formatter import format_tool_result

from .patterns import (
    VOLUME_SET_RE, VOLUME_UP_RE, VOLUME_DOWN_RE,
    PLAY_PLAYLIST_RE, PLAY_TRACK_RE, PLAY_TRACK_GENERIC,
    NP_TRACK_ID_RE, NP_SHUFFLE_RE, NP_TRACK_NAME_RE, NP_ARTIST_RE, NP_DURATION_RE,
    PLAYLIST_TRACK_RE, active_playlist,
)
from .phrases import QUEUE_PHRASES


# ── Tuneable constants ───────────────────────────────────────────────────────

PLAYLIST_FETCH_LIMIT: int  = 50    # max tracks fetched per playlist for queue view
QUEUE_FETCH_LIMIT: int     = 20    # max tracks fetched from getQueue API
SEARCH_RESULT_LIMIT: int   = 5     # max results when searching for a track
VOLUME_STEP: int           = 10    # % increase / decrease per volume-up/down command
TRACK_MATCH_THRESHOLD: float = 0.8 # min similarity score to accept a playlist track match

# ── MCP helpers ───────────────────────────────────────────────────────────────

def get_mcp_router(agent: Any) -> Any:
    """Extract the MCP router from a ChatAgent instance."""
    if hasattr(agent, "_delegate") and hasattr(agent._delegate, "mcp_router"):
        return agent._delegate.mcp_router
    return getattr(agent, "mcp_router", None)


def extract_mcp_text(payload: Any) -> str:
    """Pull plain text out of an MCP tool result payload."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    lines: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text)
    return "\n".join(lines)


def extract_first_track_id(payload: Any) -> Optional[str]:
    """Return the first Spotify track ID found in an MCP result."""
    text = extract_mcp_text(payload)
    if not text and isinstance(payload, str):
        text = payload
    match = re.search(r"ID:\s*([A-Za-z0-9]+)", text, re.IGNORECASE)
    return match.group(1) if match else None


# ── Queue ─────────────────────────────────────────────────────────────────────

async def handle_show_queue(router: Any, normalized: str) -> Optional[str]:
    """
    Show the upcoming queue accurately.

    Strategy:
      1. getNowPlaying → current track ID + shuffle state
      2. If a playlist was started via JARVIS (active_playlist cache set)
         and shuffle is OFF → getPlaylistTracks, slice from current position.
         This matches exactly what the Spotify app shows.
      3. Otherwise → fall back to getQueue API (best-effort).
    """
    if normalized not in QUEUE_PHRASES:
        return None

    np_result = await router.execute_tool("getNowPlaying")
    if np_result.is_error:
        return f"I couldn't fetch playback state: {np_result.error or 'unknown error'}"

    np_text = extract_mcp_text(np_result.result)

    current_id     = m.group(1) if (m := NP_TRACK_ID_RE.search(np_text))  else ""
    shuffle_on     = m.group(1).lower() == "on" if (m := NP_SHUFFLE_RE.search(np_text)) else False
    current_name   = m.group(1) if (m := NP_TRACK_NAME_RE.search(np_text)) else "Unknown"
    current_artist = m.group(1) if (m := NP_ARTIST_RE.search(np_text))    else "Unknown"
    current_dur    = m.group(1) if (m := NP_DURATION_RE.search(np_text))  else "?"

    now_line = f'Currently Playing: "{current_name}" by {current_artist} ({current_dur})'

    playlist_id = active_playlist["id"]
    if playlist_id and not shuffle_on:
        tracks_result = await router.execute_tool(
            "getPlaylistTracks", playlistId=playlist_id, limit=PLAYLIST_FETCH_LIMIT
        )
        if not tracks_result.is_error:
            tracks: list[dict] = []
            for line in extract_mcp_text(tracks_result.result).splitlines():
                m2 = PLAYLIST_TRACK_RE.match(line.strip())
                if m2:
                    tracks.append({
                        "name": m2.group(1), "artist": m2.group(2),
                        "duration": m2.group(3), "id": m2.group(4),
                    })
            if tracks:
                idx = next((i for i, t in enumerate(tracks) if t["id"] == current_id), None)
                upcoming = tracks[idx + 1:] if idx is not None else tracks
                if not upcoming:
                    return f"# Spotify Queue\n\n{now_line}\n\nNo more tracks in this playlist."
                formatted = "\n".join(
                    f'{i + 1}. "{t["name"]}" by {t["artist"]} ({t["duration"]}) - ID: {t["id"]}'
                    for i, t in enumerate(upcoming)
                )
                name = active_playlist["name"] or "playlist"
                return f"# Spotify Queue\n\n{now_line}\n\nNext {len(upcoming)} from \u201c{name}\u201d:\n\n{formatted}"

    fallback = await router.execute_tool("getQueue", limit=QUEUE_FETCH_LIMIT)
    if fallback.is_error:
        return f"I couldn't fetch your queue: {fallback.error or 'unknown error'}"
    return format_tool_result("getQueue", fallback.result)


# ── Volume ────────────────────────────────────────────────────────────────────

async def handle_volume(router: Any, normalized: str) -> Optional[str]:
    """Handle set-volume and relative adjust-volume commands."""
    if m := VOLUME_SET_RE.search(normalized):
        volume = max(0, min(100, int(m.group(1))))
        result = await router.execute_tool("setVolume", volumePercent=volume)
        if result.is_error:
            return f"I couldn't set Spotify volume to {volume}%: {result.error or 'unknown error'}"
        return f"Spotify volume set to {volume}%."

    if VOLUME_UP_RE.search(normalized):
        result = await router.execute_tool("adjustVolume", adjustment=VOLUME_STEP)
        if result.is_error:
            return f"I couldn't increase Spotify volume: {result.error or 'unknown error'}"
        return format_tool_result("adjustVolume", result.result)

    if VOLUME_DOWN_RE.search(normalized):
        result = await router.execute_tool("adjustVolume", adjustment=-VOLUME_STEP)
        if result.is_error:
            return f"I couldn't decrease Spotify volume: {result.error or 'unknown error'}"
        return format_tool_result("adjustVolume", result.result)

    return None


# ── Playlist playback ─────────────────────────────────────────────────────────

async def handle_play_playlist(router: Any, transcript: str) -> Optional[str]:
    """Resolve a named playlist and start playback."""
    match = PLAY_PLAYLIST_RE.match(transcript.strip())
    if not match:
        return None

    playlist_name = (match.group(1) or match.group(2) or "").strip()
    if not playlist_name:
        return "Please tell me which playlist you want to play."

    playlist = await resolve_playlist_match(router, playlist_name)
    if not playlist:
        return f'I couldn\'t find a playlist matching "{playlist_name}".'

    playlist_id = str(playlist.get("id", "")).strip()
    if not playlist_id:
        return f"I found \"{playlist.get('name', playlist_name)}\" but couldn't read its playlist ID."

    result = await router.execute_tool("playMusic", type="playlist", id=playlist_id)
    if result.is_error:
        return f"I couldn't start playlist \"{playlist.get('name', playlist_name)}\": {result.error or 'unknown error'}"

    # Cache so handle_show_queue can show accurate track order
    active_playlist["id"]   = playlist_id
    active_playlist["name"] = playlist.get("name", playlist_name)

    return f"Now playing playlist: {playlist.get('name', playlist_name)}"


# ── Track playback ────────────────────────────────────────────────────────────

async def _resolve_track_in_active_playlist(
    router: Any, query: str
) -> Optional[tuple[str, str]]:
    """
    Look up a track by name in the active playlist before falling back to
    global search. Returns (track_id, track_name) or None.
    Prefers closer length matches to avoid "Lovers" beating "Lover".
    """
    playlist_id = active_playlist["id"]
    if not playlist_id:
        return None

    result = await router.execute_tool("getPlaylistTracks", playlistId=playlist_id, limit=50)
    if result.is_error:
        return None

    q = query.lower().strip()
    best: Optional[tuple[str, str]] = None
    best_score = 0.0

    for line in extract_mcp_text(result.result).splitlines():
        m = PLAYLIST_TRACK_RE.match(line.strip())
        if not m:
            continue
        name, track_id = m.group(1), m.group(4)
        name_lower = name.lower()
        if name_lower == q:
            return (track_id, name)
        if q in name_lower or name_lower in q:
            score = len(q) / max(len(name_lower), 1)
            if score > best_score:
                best_score = score
                best = (track_id, name)

    return best if best_score >= TRACK_MATCH_THRESHOLD else None


async def handle_play_track(router: Any, transcript: str) -> Optional[str]:
    """
    Find a track and play it without destroying the queue.

    Priority: active playlist → global search
    Strategy: addToQueue + skipToNext (preserves existing queue context)
    """
    match = PLAY_TRACK_RE.match(transcript.strip())
    if not match:
        return None

    track_query = match.group(1).strip()
    if not track_query or track_query.lower() in PLAY_TRACK_GENERIC:
        return None

    playlist_match = await _resolve_track_in_active_playlist(router, track_query)
    if playlist_match:
        track_id, track_name = playlist_match
    else:
        search = await router.execute_tool("searchSpotify", query=track_query, type="track", limit=SEARCH_RESULT_LIMIT)
        if search.is_error:
            return f'I couldn\'t search for "{track_query}": {search.error or "unknown error"}'
            
        q = track_query.lower()
        best_id, best_name = None, None
        best_score = 0.0
        first_id, first_name = None, None
        
        for line in extract_mcp_text(search.result).splitlines():
            m = PLAYLIST_TRACK_RE.match(line.strip())
            if not m:
                continue
            name, tid = m.group(1), m.group(4)
            if not first_id:
                first_id, first_name = tid, name
                
            name_lower = name.lower()
            if name_lower == q:
                best_id, best_name = tid, name
                best_score = 1.0
                break
            if q in name_lower or name_lower in q:
                score = len(q) / max(len(name_lower), 1)
                if score > best_score:
                    best_score = score
                    best_id, best_name = tid, name
                    
        if best_id and best_score >= TRACK_MATCH_THRESHOLD:
            track_id, track_name = best_id, best_name
        elif first_id:
            track_id, track_name = first_id, first_name
        else:
            return f'I couldn\'t find any tracks matching "{track_query}".'

    queue_result = await router.execute_tool("addToQueue", type="track", id=track_id)
    if queue_result.is_error:
        return f'I couldn\'t queue "{track_name}": {queue_result.error or "unknown error"}'

    skip_result = await router.execute_tool("skipToNext")
    if skip_result.is_error and "Unexpected number in JSON" not in str(skip_result.error):
        return f'Queued "{track_name}" but couldn\'t skip to it: {skip_result.error or "unknown error"}'

    return f"Now playing: {track_name}"
