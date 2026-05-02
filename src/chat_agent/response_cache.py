"""
Safe LLM response cache for low-staleness, repeatable general queries.

Caches deterministic responses to reduce LLM provider calls
with conservative guardrails for freshness/staleness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from .cache_key_builder import build_cache_key
from .cache_metrics import CacheMetrics
from .eviction_policy import (
    EvictionPolicy,
    CacheInvalidator,
    find_and_remove_oldest,
    invalidate_all,
    invalidate_by_session,
    remove_entry,
)
from .guardrails import (
    CacheEligibilityDecision,
    evaluate_transcript_eligibility,
    evaluate_response_eligibility,
)


@dataclass(slots=True)
class CachedResponseEntry:
    """Cached LLM response entry."""
    key: str
    response_text: str
    created_at_epoch: float
    expires_at_epoch: float
    created_at_utc: str
    expires_at_utc: str
    source_latency_ms: float
    session_id: str
    provider: str
    model: str
    transcript_key: str
    tools_fingerprint: str


class LLMResponseCache:
    """
    TTL-based response cache with conservative cacheability guardrails.
    
    Orchestrates cache key building, eligibility evaluation, eviction, metrics.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 180,
        max_entries: int = 256,
        min_chars: int = 24,
        persistence_path: Path | None = None,
    ) -> None:
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for entries
            max_entries: Maximum cache size before LRU eviction
            min_chars: Minimum response length to cache
            persistence_path: Optional path for cache persistence
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.min_chars = min_chars
        self.persistence_path = persistence_path

        self._entries: dict[str, CachedResponseEntry] = {}
        self._session_index: dict[str, set[str]] = {}

        self._metrics = CacheMetrics()
        self._eviction_policy = EvictionPolicy(ttl_seconds, max_entries)
        self._invalidator = CacheInvalidator()

        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    @property
    def lookup_count(self) -> int:
        """Get lookup count for backward compatibility."""
        return self._metrics.lookup_count

    @property
    def hit_count(self) -> int:
        """Get hit count for backward compatibility."""
        return self._metrics.hit_count

    @property
    def miss_count(self) -> int:
        """Get miss count for backward compatibility."""
        return self._metrics.miss_count

    @property
    def skip_count(self) -> int:
        """Get skip count for backward compatibility."""
        return self._metrics.skip_count

    @property
    def stale_block_count(self) -> int:
        """Get stale block count for backward compatibility."""
        return self._metrics.stale_block_count

    @property
    def write_count(self) -> int:
        """Get write count for backward compatibility."""
        return self._metrics.write_count

    @property
    def invalidation_count(self) -> int:
        """Get invalidation count for backward compatibility."""
        return self._metrics.invalidation_count

    @property
    def eviction_count(self) -> int:
        """Get eviction count for backward compatibility."""
        return self._eviction_policy.eviction_count

    @property
    def estimated_latency_saved_ms(self) -> float:
        """Get estimated latency saved for backward compatibility."""
        return self._metrics.estimated_latency_saved_ms

    def _now_epoch(self) -> float:
        """Get current Unix timestamp."""
        return time.time()

    def _iso_from_epoch(self, value: float) -> str:
        """Convert Unix timestamp to ISO 8601 string."""
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    def record_skip(self, reason: str) -> None:
        """Record a skipped caching decision."""
        self._metrics.record_skip(reason)

    def build_key(
        self,
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
        """Build cache key from request parameters."""
        return build_cache_key(
            transcript=transcript,
            session_id=session_id,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools_payload=tools_payload,
        )

    def evaluate_eligibility(
        self,
        *,
        transcript: str,
        intent_type: str | None,
        supports_tools: bool,
        tools_payload: list[dict[str, Any]] | None,
        allow_tool_providers: bool,
        messages: list[dict[str, str]],
    ) -> CacheEligibilityDecision:
        """Evaluate if a transcript is eligible for caching."""
        return evaluate_transcript_eligibility(
            transcript=transcript,
            intent_type=intent_type,
            supports_tools=supports_tools,
            tools_payload=tools_payload,
            allow_tool_providers=allow_tool_providers,
            messages=messages,
        )

    def lookup(self, key: str) -> str | None:
        """
        Look up cached response.
        
        Returns:
            Cached response text or None if miss/expired
        """
        self._metrics.record_lookup()
        entry = self._entries.get(key)
        
        if not entry:
            self._metrics.record_miss()
            return None

        now = self._now_epoch()
        if now >= entry.expires_at_epoch:
            self._metrics.record_stale()
            self._metrics.record_miss()
            invalidation_count, self._entries = remove_entry(
                self._entries,
                self._session_index,
                key,
                count_invalidation=False,
            )
            return None

        self._metrics.record_hit(entry.source_latency_ms)
        return entry.response_text

    def should_store_response(self, response_text: str) -> CacheEligibilityDecision:
        """Evaluate if a response is eligible for caching."""
        return evaluate_response_eligibility(response_text, self.min_chars)

    def store(
        self,
        *,
        key: str,
        response_text: str,
        source_latency_ms: float,
        session_id: str,
        provider: str,
        model: str,
        transcript_key: str,
        tools_fingerprint: str,
    ) -> None:
        """Store a response in cache."""
        now = self._now_epoch()
        expiry = self._eviction_policy.calculate_expiry(now)
        
        entry = CachedResponseEntry(
            key=key,
            response_text=response_text,
            created_at_epoch=now,
            expires_at_epoch=expiry,
            created_at_utc=self._iso_from_epoch(now),
            expires_at_utc=self._iso_from_epoch(expiry),
            source_latency_ms=max(0.0, source_latency_ms),
            session_id=session_id,
            provider=provider,
            model=model,
            transcript_key=transcript_key,
            tools_fingerprint=tools_fingerprint,
        )

        # Replace if exists
        _, self._entries = remove_entry(
            self._entries,
            self._session_index,
            key,
            count_invalidation=False,
        )
        
        self._entries[key] = entry
        self._session_index.setdefault(session_id, set()).add(key)
        self._metrics.record_write()
        self._evict_if_needed()
        self._persist()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max_entries."""
        while self._eviction_policy.should_evict(len(self._entries)):
            if find_and_remove_oldest(self._entries, self._session_index):
                self._eviction_policy.record_eviction()

    def invalidate_session(self, session_id: str) -> None:
        """Invalidate all entries for a session."""
        removed_count = invalidate_by_session(
            self._entries,
            self._session_index,
            session_id,
        )
        if removed_count > 0:
            self._metrics.record_invalidation_batch(removed_count)
            self._persist()

    def invalidate_all(self) -> None:
        """Invalidate entire cache."""
        removed_count = invalidate_all(self._entries, self._session_index)
        if removed_count > 0:
            self._metrics.record_invalidation_batch(removed_count)
            self._persist()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return self._metrics.get_all_stats(
            self.ttl_seconds,
            self.max_entries,
            len(self._entries),
        )

    def _persist(self) -> None:
        """Persist cache to disk if configured."""
        if not self.persistence_path:
            return
        
        payload = {
            "config": {
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "min_chars": self.min_chars,
            },
            "stats": {
                "lookup_count": self._metrics.lookup_count,
                "hit_count": self._metrics.hit_count,
                "miss_count": self._metrics.miss_count,
                "skip_count": self._metrics.skip_count,
                "skip_reasons": self._metrics._skip_reasons,
                "stale_block_count": self._metrics.stale_block_count,
                "write_count": self._metrics.write_count,
                "invalidation_count": self._metrics.invalidation_count,
                "eviction_count": self._eviction_policy.eviction_count,
                "estimated_latency_saved_ms": self._metrics.estimated_latency_saved_ms,
            },
            "entries": [asdict(entry) for entry in self._entries.values()],
        }
        self.persistence_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            raw = json.loads(self.persistence_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        now = self._now_epoch()
        for item in raw.get("entries", []):
            try:
                entry = CachedResponseEntry(
                    key=str(item["key"]),
                    response_text=str(item.get("response_text", "")),
                    created_at_epoch=float(item.get("created_at_epoch", now)),
                    expires_at_epoch=float(item.get("expires_at_epoch", now)),
                    created_at_utc=str(item.get("created_at_utc", self._iso_from_epoch(now))),
                    expires_at_utc=str(item.get("expires_at_utc", self._iso_from_epoch(now))),
                    source_latency_ms=float(item.get("source_latency_ms", 0.0)),
                    session_id=str(item.get("session_id", "default")),
                    provider=str(item.get("provider", "")),
                    model=str(item.get("model", "")),
                    transcript_key=str(item.get("transcript_key", "")),
                    tools_fingerprint=str(item.get("tools_fingerprint", "none")),
                )
            except (KeyError, TypeError, ValueError):
                continue
            
            # Skip expired entries
            if now >= entry.expires_at_epoch:
                continue
            
            self._entries[entry.key] = entry
            self._session_index.setdefault(entry.session_id, set()).add(entry.key)

        # Restore metrics from stats
        stats = raw.get("stats", {})
        self._metrics.lookup_count = int(stats.get("lookup_count", 0))
        self._metrics.hit_count = int(stats.get("hit_count", 0))
        self._metrics.miss_count = int(stats.get("miss_count", 0))
        self._metrics.skip_count = int(stats.get("skip_count", 0))
        self._metrics._skip_reasons = {
            str(k): int(v) for k, v in dict(stats.get("skip_reasons", {})).items()
        }
        self._metrics.stale_block_count = int(stats.get("stale_block_count", 0))
        self._metrics.write_count = int(stats.get("write_count", 0))
        self._metrics.invalidation_count = int(stats.get("invalidation_count", 0))
        self._eviction_policy.eviction_count = int(stats.get("eviction_count", 0))
        self._metrics.estimated_latency_saved_ms = float(
            stats.get("estimated_latency_saved_ms", 0.0)
        )
