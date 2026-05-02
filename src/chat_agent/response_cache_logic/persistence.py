"""
Cache Persistence Logic

Handles loading and saving of the LLM response cache.
"""

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from ..models import CachedResponseEntry

if TYPE_CHECKING:
    from ..response_cache import LLMResponseCache


def persist_cache(cache: 'LLMResponseCache') -> None:
    """Persist cache to disk if configured."""
    if not cache.persistence_path:
        return

    payload = {
        "config": {
            "ttl_seconds": cache.ttl_seconds,
            "max_entries": cache.max_entries,
            "min_chars": cache.min_chars,
        },
        "stats": {
            "lookup_count": cache._metrics.lookup_count,
            "hit_count": cache._metrics.hit_count,
            "miss_count": cache._metrics.miss_count,
            "skip_count": cache._metrics.skip_count,
            "skip_reasons": cache._metrics._skip_reasons,
            "stale_block_count": cache._metrics.stale_block_count,
            "write_count": cache._metrics.write_count,
            "invalidation_count": cache._metrics.invalidation_count,
            "eviction_count": cache._eviction_policy.eviction_count,
            "estimated_latency_saved_ms": cache._metrics.estimated_latency_saved_ms,
        },
        "entries": [asdict(entry) for entry in cache._entries.values()],
    }
    cache.persistence_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_cache(cache: 'LLMResponseCache') -> None:
    """Load cache from disk if it exists."""
    if not cache.persistence_path or not cache.persistence_path.exists():
        return

    try:
        raw = json.loads(cache.persistence_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    now = cache._now_epoch()
    for item in raw.get("entries", []):
        try:
            entry = CachedResponseEntry(
                key=str(item["key"]),
                response_text=str(item.get("response_text", "")),
                created_at_epoch=float(item.get("created_at_epoch", now)),
                expires_at_epoch=float(item.get("expires_at_epoch", now)),
                created_at_utc=str(item.get("created_at_utc", cache._iso_from_epoch(now))),
                expires_at_utc=str(item.get("expires_at_utc", cache._iso_from_epoch(now))),
                source_latency_ms=float(item.get("source_latency_ms", 0.0)),
                session_id=str(item.get("session_id", "default")),
                provider=str(item.get("provider", "")),
                model=str(item.get("model", "")),
                transcript_key=str(item.get("transcript_key", "")),
                tools_fingerprint=str(item.get("tools_fingerprint", "none")),
            )
        except (KeyError, TypeError, ValueError):
            continue

        if now >= entry.expires_at_epoch:
            continue

        cache._entries[entry.key] = entry
        cache._session_index.setdefault(entry.session_id, set()).add(entry.key)

    stats = raw.get("stats", {})
    cache._metrics.lookup_count = int(stats.get("lookup_count", 0))
    cache._metrics.hit_count = int(stats.get("hit_count", 0))
    cache._metrics.miss_count = int(stats.get("miss_count", 0))
    cache._metrics.skip_count = int(stats.get("skip_count", 0))
    cache._metrics._skip_reasons = {
        str(k): int(v) for k, v in dict(stats.get("skip_reasons", {})).items()
    }
    cache._metrics.stale_block_count = int(stats.get("stale_block_count", 0))
    cache._metrics.write_count = int(stats.get("write_count", 0))
    cache._metrics.invalidation_count = int(stats.get("invalidation_count", 0))
    cache._eviction_policy.eviction_count = int(stats.get("eviction_count", 0))
    cache._metrics.estimated_latency_saved_ms = float(
        stats.get("estimated_latency_saved_ms", 0.0)
    )
