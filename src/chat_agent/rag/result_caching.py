"""
rag/result_caching.py
──────────────────────
Cache Spotify tool results into the vector store for future RAG retrieval.
Safe no-op if the vector store is unavailable.
"""
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Payload coercion helpers ──────────────────────────────────────────────────

def _coerce_payload(tool_result: Any) -> Any:
    if isinstance(tool_result, str):
        text = tool_result.strip()
        if text and text[0] in "{[":
            try:
                return json.loads(text)
            except Exception:
                return tool_result
    return tool_result


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_spotify_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    parts = text.rsplit(":", 1)
    if len(parts) == 2 and text.startswith("spotify:"):
        return parts[1]
    return text


# ── Field extractors ──────────────────────────────────────────────────────────

def _extract_artist(entry: dict[str, Any]) -> str:
    artist = entry.get("artist")
    if isinstance(artist, str) and artist.strip():
        return artist.strip()
    artists = entry.get("artists")
    if isinstance(artists, list):
        names = []
        for item in artists:
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
            elif isinstance(item, str) and item.strip():
                names.append(item.strip())
        if names:
            return ", ".join(names)
    return ""


def _extract_track(entry: dict[str, Any]) -> dict[str, str]:
    track = entry.get("track") if isinstance(entry.get("track"), dict) else entry
    if not isinstance(track, dict):
        return {}
    track_id = _first_non_empty(track.get("id"), track.get("spotify_id"), track.get("uri"))
    name = _first_non_empty(track.get("name"), track.get("title"))
    artist = _extract_artist(track)
    if not name or not track_id:
        return {}
    return {"id": _normalize_spotify_id(track_id), "name": name, "artist": artist}


def _extract_playlists(tool_result: Any) -> list[dict[str, Any]]:
    payload = _coerce_payload(tool_result)
    if isinstance(payload, dict):
        playlists = payload.get("playlists")
        if isinstance(playlists, list):
            return [p for p in playlists if isinstance(p, dict)]
        if isinstance(payload.get("items"), list):
            return [p for p in payload["items"] if isinstance(p, dict)]
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    return []


def _extract_tracks(tool_result: Any) -> list[dict[str, Any]]:
    payload = _coerce_payload(tool_result)
    if isinstance(payload, dict):
        for key in ("tracks", "items", "queue", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        current = payload.get("currently_playing") or payload.get("currentlyPlaying")
        if isinstance(current, dict):
            return [current]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


# ── Vector store writers ──────────────────────────────────────────────────────

def _cache_tracks(retriever: Any, items: list[dict[str, Any]], ttl_hours: int) -> None:
    for item in items[:25]:
        normalized = _extract_track(item)
        if not normalized:
            continue
        retriever.cache_track(
            track_id=normalized["id"],
            track_name=normalized["name"],
            artist=normalized.get("artist", ""),
            ttl_hours=ttl_hours,
        )


def _cache_playlists(retriever: Any, items: list[dict[str, Any]], ttl_hours: int) -> None:
    for playlist in items[:20]:
        playlist_id = _normalize_spotify_id(
            _first_non_empty(playlist.get("id"), playlist.get("spotify_id"), playlist.get("uri"))
        )
        playlist_name = _first_non_empty(playlist.get("name"), playlist.get("title"))
        if not playlist_id or not playlist_name:
            continue
        retriever.cache_playlist(
            playlist_id=playlist_id,
            playlist_name=playlist_name,
            description=_first_non_empty(playlist.get("description")),
            ttl_hours=ttl_hours,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def cache_tool_results(
    tool_name: str,
    tool_result: Any,
    user_query: Optional[str] = None,
) -> None:
    """
    Cache Spotify tool results in the vector store for future RAG retrieval.
    Safe no-op if the vector store is unavailable.
    """
    try:
        from . import get_rag_retriever
        retriever = get_rag_retriever()
        payload = _coerce_payload(tool_result)

        if tool_name == "getMyPlaylists":
            _cache_playlists(retriever, _extract_playlists(payload), ttl_hours=1)

        elif tool_name in {"getQueue", "getPlaylistTracks", "getNowPlaying", "getTopTracks"}:
            ttl = 2 if tool_name == "getQueue" else 6
            _cache_tracks(retriever, _extract_tracks(payload), ttl_hours=ttl)

        elif tool_name == "searchSpotify":
            results: list[dict] = []
            if isinstance(payload, dict):
                maybe = payload.get("results")
                if isinstance(maybe, list):
                    results = [i for i in maybe if isinstance(i, dict)]
            elif isinstance(payload, list):
                results = [i for i in payload if isinstance(i, dict)]

            tracks = [i for i in results if str(i.get("type", "")).lower() in {"track", "song", ""}]
            playlists = [i for i in results if str(i.get("type", "")).lower() == "playlist"]
            _cache_tracks(retriever, tracks, ttl_hours=6)
            _cache_playlists(retriever, playlists, ttl_hours=1)

        elif tool_name == "checkSpotifyAuth":
            if not isinstance(payload, dict):
                payload = {}
            user = payload.get("user", {})
            retriever.cache_user_profile(
                user_id=user.get("id", ""),
                display_name=user.get("display_name", ""),
                top_genres=payload.get("top_genres", []),
                product=user.get("product", "free"),
            )

        if user_query:
            retriever.vector_store.log_user_action(
                query=user_query,
                tool_name=tool_name,
                result_type=type(tool_result).__name__,
            )

    except Exception as e:
        logger.debug(f"Failed to cache tool results: {e}")
