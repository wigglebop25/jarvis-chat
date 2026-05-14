"""
Smart music handler for queue-aware playback.

When user requests to play a track, checks if it's already in the queue.
If it is, skips to that position. If not, searches and plays it.
"""

import re
from typing import Any, Optional
from .mcp.router_protocol import MCPRouterLike
from .mcp.models import MCPToolResult


async def play_track_smart(
    router: MCPRouterLike,
    track_name: str,
    device_id: Optional[str] = None,
) -> MCPToolResult:
    """
    Play a track intelligently:
    1. Get current queue
    2. Search for track name in queue
    3. If found, skip to it (preserves context)
    4. If not found, search Spotify and play it
    
    Args:
        router: MCP router for tool execution
        track_name: Name of the track to play
        device_id: Optional Spotify device ID
        
    Returns:
        MCPToolResult with playback confirmation
    """
    try:
        # Step 1: Get current queue
        queue_result = await router.execute_tool("getQueue", limit=50)
        if queue_result.is_error:
            # Fallback: just play the track by searching
            return await _search_and_play(router, track_name, device_id)
        
        # Step 2: Parse queue and search for the track
        queue_data = queue_result.result
        track_position = _find_track_in_queue(track_name, queue_data)
        
        if track_position is not None and track_position > 0:
            # Track is in queue, skip to it
            return await _skip_to_position(router, track_position, device_id)
        
        # Step 3: Track not in queue, search and play
        return await _search_and_play(router, track_name, device_id)
        
    except Exception as e:
        return MCPToolResult(
            is_error=True,
            result=f"Error playing track '{track_name}': {str(e)}"
        )


def _find_track_in_queue(track_name: str, queue_data: Any) -> Optional[int]:
    """
    Search for a track in the queue by name.
    
    Returns:
        Position in queue (starting at 1), or None if not found
    """
    if isinstance(queue_data, str):
        # Parse text-based queue
        lines = queue_data.split('\n')
        query_normalized = _normalize_name(track_name)
        
        for line in lines:
            # Look for pattern: "N. Track Name — Artist"
            match = re.match(r'^\s*(\d+)\.\s+"?([^"—•-]+)', line)
            if match:
                position = int(match.group(1))
                name = match.group(2).strip()
                if _normalize_name(name) == query_normalized:
                    return position
                    
    elif isinstance(queue_data, dict):
        # Parse structured queue (dict with queue list)
        items = queue_data.get("items") or queue_data.get("queue") or []
        query_normalized = _normalize_name(track_name)
        
        for idx, item in enumerate(items, start=1):
            if isinstance(item, dict):
                name = item.get("name", "")
                if _normalize_name(name) == query_normalized:
                    return idx
                    
    elif isinstance(queue_data, list):
        # Direct list of queue items
        query_normalized = _normalize_name(track_name)
        for idx, item in enumerate(queue_data, start=1):
            if isinstance(item, dict):
                name = item.get("name", "")
            else:
                name = str(item)
            if _normalize_name(name) == query_normalized:
                return idx
    
    return None


def _normalize_name(name: str) -> str:
    """Normalize a track/artist name for comparison."""
    # Remove special characters, convert to lowercase
    name = re.sub(r"[^a-z0-9\s]+", "", name.lower())
    # Remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()
    return name


async def _skip_to_position(
    router: MCPRouterLike,
    position: int,
    device_id: Optional[str] = None,
) -> MCPToolResult:
    """Skip to a specific position in the queue."""
    try:
        # Skip forward/backward to reach the target position
        # For now, this is a simplified version - just skip once
        # In production, would need to track current position and skip multiple times
        if position == 1:
            # Already at/near current track
            return MCPToolResult(result="Already playing from this position")
        
        for _ in range(position - 1):
            await router.execute_tool("skipToNext", deviceId=device_id or "")
        
        return MCPToolResult(result=f"Skipped to position {position} in queue")
    except Exception as e:
        return MCPToolResult(
            is_error=True,
            result=f"Error skipping to position: {str(e)}"
        )


async def _search_and_play(
    router: MCPRouterLike,
    track_name: str,
    device_id: Optional[str] = None,
) -> MCPToolResult:
    """Search for a track and play it."""
    try:
        # Search for the track
        search_result = await router.execute_tool(
            "searchSpotify",
            query=track_name,
            type="track",
        )
        
        if search_result.is_error:
            return search_result
        
        # Parse search results to get first track ID
        track_id = _extract_first_track_id(search_result.result)
        if not track_id:
            return MCPToolResult(
                is_error=True,
                result=f"Could not find track: {track_name}"
            )
        
        # Play the track
        play_result = await router.execute_tool(
            "playMusic",
            id=track_id,
            type="track",
            deviceId=device_id or "",
        )
        
        return play_result
        
    except Exception as e:
        return MCPToolResult(
            is_error=True,
            result=f"Error searching and playing: {str(e)}"
        )


def _extract_first_track_id(search_results: Any) -> Optional[str]:
    """Extract the first track ID from search results."""
    if isinstance(search_results, str):
        # Parse text-based results
        # Look for pattern: ID: xxxxx or similar
        match = re.search(r"ID:\s*([A-Za-z0-9]+)", search_results)
        if match:
            return match.group(1)
    
    elif isinstance(search_results, dict):
        tracks = search_results.get("tracks", [])
        if tracks and isinstance(tracks, list) and len(tracks) > 0:
            first = tracks[0]
            if isinstance(first, dict):
                return first.get("id")
    
    elif isinstance(search_results, list):
        if len(search_results) > 0:
            first = search_results[0]
            if isinstance(first, dict):
                return first.get("id")
    
    return None
