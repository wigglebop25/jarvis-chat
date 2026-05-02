"""
Cache key building and fingerprinting utilities.

Generates deterministic cache keys from transcript, config, and tool payloads.
"""

import hashlib
import json
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for cache key comparison."""
    return " ".join((text or "").strip().lower().split())


def tools_fingerprint(tools_payload: list[dict[str, Any]] | None) -> str:
    """
    Generate fingerprint for tool definitions.
    
    Args:
        tools_payload: Tool definitions or None
    
    Returns:
        SHA256 hash of canonical tool payload, or "none" if empty
    """
    if not tools_payload:
        return "none"
    canonical = json.dumps(
        tools_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_cache_key(
    *,
    transcript: str,
    session_id: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    tools_payload: list[dict[str, Any]] | None,
) -> tuple[str, str, str]:
    """
    Build cache key from transcript and configuration.
    
    Returns:
        Tuple of (cache_key, transcript_key, tools_fingerprint)
    """
    transcript_key = normalize_text(transcript)
    tools_fp = tools_fingerprint(tools_payload)
    
    key_payload = {
        "v": 1,
        "session_id": session_id,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt_hash": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "transcript_key": transcript_key,
        "tools_fingerprint": tools_fp,
    }
    
    canonical = json.dumps(
        key_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    cache_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return cache_key, transcript_key, tools_fp
