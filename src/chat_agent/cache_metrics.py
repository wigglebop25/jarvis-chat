"""
Cache metrics and analytics.

Tracks hits, misses, skips, and performance metrics.
"""

from typing import Any


class CacheMetrics:
    """Tracks cache performance and operation metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.lookup_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.skip_count = 0
        self.stale_block_count = 0
        self.write_count = 0
        self.invalidation_count = 0
        self.eviction_count = 0
        self.estimated_latency_saved_ms = 0.0
        self._skip_reasons: dict[str, int] = {}
    
    def record_lookup(self) -> None:
        """Record one lookup attempt."""
        self.lookup_count += 1
    
    def record_hit(self, latency_ms: float) -> None:
        """
        Record cache hit.
        
        Args:
            latency_ms: Latency saved by hit
        """
        self.hit_count += 1
        self.estimated_latency_saved_ms += max(0.0, latency_ms)
    
    def record_miss(self) -> None:
        """Record cache miss."""
        self.miss_count += 1
    
    def record_stale(self) -> None:
        """Record stale entry check."""
        self.stale_block_count += 1
    
    def record_skip(self, reason: str) -> None:
        """
        Record skipped caching attempt.
        
        Args:
            reason: Reason for skip
        """
        self.skip_count += 1
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1
    
    def record_write(self) -> None:
        """Record one cache write."""
        self.write_count += 1
    
    def record_invalidation(self) -> None:
        """Record one invalidation."""
        self.invalidation_count += 1
    
    def record_invalidation_batch(self, count: int) -> None:
        """
        Record multiple invalidations.
        
        Args:
            count: Number of entries invalidated
        """
        self.invalidation_count += count
    
    def record_eviction(self) -> None:
        """Record one eviction."""
        self.eviction_count += 1
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Hit rate between 0.0 and 1.0
        """
        if self.lookup_count == 0:
            return 0.0
        return self.hit_count / self.lookup_count
    
    def get_all_stats(
        self,
        ttl_seconds: int,
        max_entries: int,
        entry_count: int,
    ) -> dict[str, Any]:
        """
        Get comprehensive stats dictionary.
        
        Args:
            ttl_seconds: Cache TTL configuration
            max_entries: Max entries configuration
            entry_count: Current number of entries
        
        Returns:
            Dictionary of stats
        """
        hit_rate = self.get_hit_rate()
        return {
            "enabled": True,
            "ttl_seconds": ttl_seconds,
            "max_entries": max_entries,
            "entry_count": entry_count,
            "lookup_count": self.lookup_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "skip_count": self.skip_count,
            "skip_reasons": dict(sorted(self._skip_reasons.items())),
            "stale_block_count": self.stale_block_count,
            "write_count": self.write_count,
            "invalidation_count": self.invalidation_count,
            "eviction_count": self.eviction_count,
            "estimated_latency_saved_ms": round(self.estimated_latency_saved_ms, 2),
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.lookup_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.skip_count = 0
        self.stale_block_count = 0
        self.write_count = 0
        self.invalidation_count = 0
        self.eviction_count = 0
        self.estimated_latency_saved_ms = 0.0
        self._skip_reasons.clear()
