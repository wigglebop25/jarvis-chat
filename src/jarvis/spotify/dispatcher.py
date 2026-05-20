"""
src/jarvis/spotify/dispatcher.py
─────────────────────────────────
Main fast-path dispatcher for deterministic Spotify commands.

Intercepts user input before it reaches the LLM. Returns a response string
if handled, or None to let the LangGraph agent take over.
"""

from typing import Optional

from chat_agent import ChatAgent
from chat_agent.tools.formatter import format_tool_result

from .handlers import (
    get_mcp_router,
    handle_show_queue,
    handle_volume,
    handle_play_playlist,
    handle_play_track,
)
from .phrases import EXACT_DISPATCH


async def try_spotify_fast_path(agent: ChatAgent, transcript: str) -> Optional[str]:
    """
    Handle deterministic Spotify commands without invoking the LLM.

    Pipeline:
      1. Show-queue handler          — getNowPlaying + getPlaylistTracks (accurate)
      2. Exact-match dispatch table  — zero-argument or fixed-argument tools
      3. Volume handler              — regex numeric extraction
      4. Play-playlist handler       — name resolution + playback
      5. Play-track handler          — addToQueue + skipToNext

    Returns None if the transcript doesn't match any fast-path pattern,
    signalling the caller to fall through to the LangGraph agent.
    """
    router = get_mcp_router(agent)
    if router is None:
        return None

    normalized = " ".join(transcript.lower().strip().split())

    # 1. Queue display (accurate — reads full playlist, not buffered API)
    if response := await handle_show_queue(router, normalized):
        return response

    # 2. Exact-match dispatch (data-driven — see phrases.EXACT_DISPATCH)
    for phrase_set, tool_name, kwargs in EXACT_DISPATCH:
        if normalized in phrase_set:
            result = await router.execute_tool(tool_name, **kwargs)
            if result.is_error:
                return f"I couldn't run {tool_name}: {result.error or 'unknown error'}"
            return format_tool_result(tool_name, result.result)

    # 3. Volume commands (require numeric extraction via regex)
    if response := await handle_volume(router, normalized):
        return response

    # 4. Play a named playlist (more specific — checked before generic track)
    if response := await handle_play_playlist(router, transcript):
        return response

    # 5. Play a specific track (addToQueue + skipToNext, preserves queue)
    if response := await handle_play_track(router, transcript):
        return response

    return None
