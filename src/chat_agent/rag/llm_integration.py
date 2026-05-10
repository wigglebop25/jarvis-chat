"""Integration module for RAG augmentation in LLM requests."""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
    match = text.rsplit(":", 1)
    if len(match) == 2 and text.startswith("spotify:"):
        return match[1]
    return text


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

    track_id = _first_non_empty(
        track.get("id"),
        track.get("spotify_id"),
        track.get("uri"),
    )
    name = _first_non_empty(track.get("name"), track.get("title"))
    artist = _extract_artist(track)
    if not name or not track_id:
        return {}

    return {
        "id": _normalize_spotify_id(track_id),
        "name": name,
        "artist": artist,
    }


def _extract_playlists(tool_result: Any) -> list[dict[str, Any]]:
    payload = _coerce_payload(tool_result)
    if isinstance(payload, dict):
        playlists = payload.get("playlists")
        if isinstance(playlists, list):
            return [pl for pl in playlists if isinstance(pl, dict)]
        if isinstance(payload.get("items"), list):
            return [pl for pl in payload["items"] if isinstance(pl, dict)]
    if isinstance(payload, list):
        return [pl for pl in payload if isinstance(pl, dict)]
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


def _cache_tracks_from_items(retriever: Any, items: list[dict[str, Any]], ttl_hours: int) -> None:
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


def _cache_playlists_from_items(retriever: Any, items: list[dict[str, Any]], ttl_hours: int) -> None:
    for playlist in items[:20]:
        playlist_id = _normalize_spotify_id(_first_non_empty(playlist.get("id"), playlist.get("spotify_id"), playlist.get("uri")))
        playlist_name = _first_non_empty(playlist.get("name"), playlist.get("title"))
        if not playlist_id or not playlist_name:
            continue
        retriever.cache_playlist(
            playlist_id=playlist_id,
            playlist_name=playlist_name,
            description=_first_non_empty(playlist.get("description")),
            ttl_hours=ttl_hours,
        )


def augment_messages_with_rag(
    messages: list[dict[str, str]],
    detected_intent: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Augment message list with RAG context before sending to LLM.
    
    Safely integrates RAG without breaking if vector store fails.
    
    Args:
        messages: Original message list
        detected_intent: Optional detected intent to guide retrieval
        
    Returns:
        Messages with augmented system prompt (or original if RAG fails)
    """
    try:
        from ..rag import get_rag_retriever
        
        retriever = get_rag_retriever()
        
        # Find the user query (usually the last message)
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break
        
        if not query:
            logger.debug("No user query found, skipping RAG augmentation")
            return messages
        
        # Retrieve context (extremely conservative for token savings)
        context = retriever.retrieve_context(query, intent=detected_intent, top_k=1)
        if not any(context.values()):
            logger.debug("No RAG context retrieved")
            return messages
        
        # Format context for injection (compact mode)
        context_str = retriever.format_context_for_prompt(context).strip()
        if not context_str:
            logger.debug("Failed to format RAG context")
            return messages
        
        # Find system message and augment it
        augmented = []
        system_found = False
        
        for msg in messages:
            if msg.get("role") == "system" and not system_found:
                # Augment system message with RAG context
                original_system = msg.get("content", "")
                augmented_system = f"{original_system}\n\n[CONTEXT FROM MEMORY]\n{context_str}"
                augmented.append({
                    "role": "system",
                    "content": augmented_system,
                })
                system_found = True
            else:
                augmented.append(msg)
        
        # If no system message, add one with context
        if not system_found:
            augmented.insert(0, {
                "role": "system",
                "content": f"[CONTEXT FROM MEMORY]\n{context_str}",
            })
        
        logger.debug(f"RAG augmented prompt with {len(context_str)} chars of context")
        return augmented
        
    except Exception as e:
        logger.debug(f"RAG augmentation failed (graceful fallback): {e}")
        # Return original messages if RAG fails - never break the LLM call
        return messages


def cache_tool_results(
    tool_name: str,
    tool_result: Any,
    user_query: Optional[str] = None,
) -> None:
    """
    Cache tool results in vector store for future RAG retrieval.
    
    Safe no-op if vector store is unavailable.
    
    Args:
        tool_name: Name of the tool that was called
        tool_result: Result from the tool
        user_query: Optional original user query context
    """
    try:
        from ..rag import get_rag_retriever
        
        retriever = get_rag_retriever()
        payload = _coerce_payload(tool_result)

        if tool_name == "getMyPlaylists":
            _cache_playlists_from_items(retriever, _extract_playlists(payload), ttl_hours=1)

        elif tool_name in {"getQueue", "getPlaylistTracks", "getNowPlaying", "getTopTracks"}:
            _cache_tracks_from_items(retriever, _extract_tracks(payload), ttl_hours=6 if tool_name != "getQueue" else 2)

        elif tool_name == "searchSpotify":
            results = []
            if isinstance(payload, dict):
                maybe_results = payload.get("results")
                if isinstance(maybe_results, list):
                    results = [item for item in maybe_results if isinstance(item, dict)]
            elif isinstance(payload, list):
                results = [item for item in payload if isinstance(item, dict)]

            track_items = []
            playlist_items = []
            for item in results:
                item_type = str(item.get("type", "")).lower()
                if item_type == "playlist":
                    playlist_items.append(item)
                elif item_type in {"track", "song", ""}:
                    track_items.append(item)
            _cache_tracks_from_items(retriever, track_items, ttl_hours=6)
            _cache_playlists_from_items(retriever, playlist_items, ttl_hours=1)

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
        
        # Log user action for mood learning
        if user_query:
            retriever.vector_store.log_user_action(
                query=user_query,
                tool_name=tool_name,
                result_type=type(tool_result).__name__,
            )
    
    except Exception as e:
        logger.debug(f"Failed to cache tool results: {e}")
        # Silently fail - never break the main flow


async def augment_messages_with_rag_async(
    messages: list[dict[str, str]],
    detected_intent: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Async version of RAG augmentation (for now, just calls sync version).
    
    In future, can implement parallel retrieval while other async operations run.
    """
    # For now, call sync version (RAG is fast enough: 5-10ms)
    return augment_messages_with_rag(messages, detected_intent)
