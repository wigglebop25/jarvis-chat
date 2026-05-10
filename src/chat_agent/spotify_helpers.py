import re
from typing import Any, Optional


_PLAYLIST_NAME_CLEANERS = ("music", "in the", "in", "the", "playlist")
_PLAYLIST_LINE_RE = re.compile(
    r'^\s*\d+\.\s+"(?P<name>.*?)"\s+\((?P<tracks>\d+)\s+tracks\)\s+-\s+ID:\s+(?P<id>[A-Za-z0-9]+)',
    re.IGNORECASE | re.MULTILINE,
)


def extract_playlist_name(text: str) -> str:
    cleaned = text.lower().split("play", 1)[1]
    for filler in _PLAYLIST_NAME_CLEANERS:
        cleaned = re.sub(rf"\b{re.escape(filler)}\b", " ", cleaned)
    return " ".join(cleaned.split()).strip()


def playlist_track_count(playlist: dict[str, Any]) -> int:
    tracks = playlist.get("tracks")
    if isinstance(tracks, dict):
        total = tracks.get("total")
        if isinstance(total, int):
            return total
        if isinstance(total, str) and total.isdigit():
            return int(total)
    if isinstance(tracks, int):
        return tracks
    return 0


def _playlist_count_from_text(text: str) -> int:
    if "doesn't have any tracks" in text.lower():
        return 0

    match = re.search(
        r"Tracks in Playlist \(\d+-\d+ of (\d+)\)",
        text,
        re.IGNORECASE,
    )
    if match:
        total = int(match.group(1))
        if total > 0:
            return total

    match = re.search(r"\*\*Tracks\*\*:\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def _normalize_playlist_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


async def resolve_playlist_match(mcp_router: Any, playlist_name: str) -> Optional[dict[str, Any]]:
    result = await mcp_router.execute_tool("getMyPlaylists", limit=50)
    if result.is_error:
        return None

    payload = result.result
    playlists: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        playlists = payload.get("playlists") or []
    elif isinstance(payload, list):
        playlists = payload
    elif isinstance(payload, str):
        playlists = [
            {
                "name": match.group("name"),
                "id": match.group("id"),
                "tracks": {"total": int(match.group("tracks"))},
            }
            for match in _PLAYLIST_LINE_RE.finditer(payload)
        ]

    if not playlists:
        return None

    target = playlist_name.strip().lower()
    
    # Handle index-based matching (e.g., #5 or just 5)
    index_match = re.search(r"^#?(\d+)$", target)
    if index_match:
        index = int(index_match.group(1)) - 1 # 1-based to 0-based
        if 0 <= index < len(playlists):
            return playlists[index]
    normalized_target = _normalize_playlist_text(playlist_name)
    exact = next(
        (pl for pl in playlists if str(pl.get("name", "")).strip().lower() == target),
        None,
    )
    if exact:
        return exact

    for playlist in playlists:
        name = str(playlist.get("name", "")).strip().lower()
        normalized_name = _normalize_playlist_text(str(playlist.get("name", "")))
        if name and (target in name or name in target):
            return playlist
        if normalized_name and (
            normalized_target in normalized_name or normalized_name in normalized_target
        ):
            return playlist

    return None


async def resolve_playlist_track_count(
    mcp_router: Any,
    playlist: dict[str, Any],
) -> int:
    count = playlist_track_count(playlist)
    if count > 0:
        return count

    playlist_id = playlist.get("id")
    if not playlist_id:
        return count

    result = await mcp_router.execute_tool("getPlaylistTracks", playlistId=playlist_id, limit=1, offset=0)
    if result.is_error:
        return count

    payload = result.result
    if isinstance(payload, str):
        verified = _playlist_count_from_text(payload)
        if verified > 0:
            return verified

        if "this playlist doesn't have any tracks" in payload.lower():
            return 0

    if isinstance(payload, dict):
        total = payload.get("total")
        if isinstance(total, int) and total > 0:
            return total

    return count
