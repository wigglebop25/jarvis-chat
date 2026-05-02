"""
Cache eviction and expiration policy.

Manages TTL enforcement, eviction by age, and session invalidation.
"""

from typing import Any


class EvictionPolicy:
    """Manages cache entry lifecycle and eviction."""
    
    def __init__(self, ttl_seconds: int, max_entries: int):
        """
        Initialize eviction policy.
        
        Args:
            ttl_seconds: Time-to-live for cache entries
            max_entries: Maximum entries before LRU eviction
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.eviction_count = 0
    
    def calculate_expiry(self, now_epoch: float) -> float:
        """
        Calculate expiration time for new entry.
        
        Args:
            now_epoch: Current Unix timestamp
        
        Returns:
            Expiration Unix timestamp
        """
        return now_epoch + self.ttl_seconds
    
    def is_expired(self, expires_at_epoch: float, now_epoch: float) -> bool:
        """
        Check if entry has expired.
        
        Args:
            expires_at_epoch: Expiration timestamp
            now_epoch: Current Unix timestamp
        
        Returns:
            True if expired
        """
        return now_epoch >= expires_at_epoch
    
    def should_evict(self, entry_count: int) -> bool:
        """
        Check if eviction is needed.
        
        Args:
            entry_count: Current number of cache entries
        
        Returns:
            True if entry count exceeds max
        """
        return entry_count > self.max_entries
    
    def get_eviction_count(self) -> int:
        """Get total eviction count."""
        return self.eviction_count
    
    def record_eviction(self) -> None:
        """Record one eviction event."""
        self.eviction_count += 1


class CacheInvalidator:
    """Handles cache invalidation and removal."""
    
    def __init__(self):
        """Initialize invalidator."""
        self.invalidation_count = 0
    
    def record_invalidation(self) -> None:
        """Record one invalidation event."""
        self.invalidation_count += 1
    
    def record_invalidation_batch(self, count: int) -> None:
        """
        Record multiple invalidation events.
        
        Args:
            count: Number of entries invalidated
        """
        self.invalidation_count += count
    
    def get_invalidation_count(self) -> int:
        """Get total invalidation count."""
        return self.invalidation_count
    
    def reset_count(self) -> None:
        """Reset invalidation counter."""
        self.invalidation_count = 0


def remove_entry(
    entries: dict[str, Any],
    session_index: dict[str, set[str]],
    key: str,
    count_invalidation: bool = True,
) -> tuple[int, dict[str, Any]]:
    """
    Remove entry from cache.
    
    Args:
        entries: Cache entries dict
        session_index: Session-to-keys index
        key: Entry key to remove
        count_invalidation: Whether to count this as invalidation
    
    Returns:
        Tuple of (invalidation_count, updated_entries)
    """
    entry = entries.pop(key, None)
    invalidation_count = 0
    
    if not entry:
        return invalidation_count, entries
    
    if count_invalidation:
        invalidation_count = 1
    
    # Update session index
    session_id = entry.get("session_id")
    if session_id:
        session_keys = session_index.get(session_id)
        if session_keys:
            session_keys.discard(key)
            if not session_keys:
                session_index.pop(session_id, None)
    
    return invalidation_count, entries


def invalidate_by_session(
    entries: dict[str, Any],
    session_index: dict[str, set[str]],
    session_id: str,
) -> int:
    """
    Invalidate all entries for a session.
    
    Args:
        entries: Cache entries dict
        session_index: Session-to-keys index
        session_id: Session ID to invalidate
    
    Returns:
        Number of entries removed
    """
    keys_to_remove = list(session_index.get(session_id, set()))
    removed = 0
    
    for key in keys_to_remove:
        if key in entries:
            entries.pop(key)
            removed += 1
    
    # Clean up session index
    session_index.pop(session_id, None)
    
    return removed


def invalidate_all(
    entries: dict[str, Any],
    session_index: dict[str, set[str]],
) -> int:
    """
    Invalidate entire cache.
    
    Args:
        entries: Cache entries dict
        session_index: Session-to-keys index
    
    Returns:
        Number of entries removed
    """
    count = len(entries)
    entries.clear()
    session_index.clear()
    return count


def find_and_remove_oldest(
    entries: dict[str, Any],
    session_index: dict[str, set[str]],
) -> bool:
    """
    Find and remove oldest entry by creation timestamp.
    
    Args:
        entries: Cache entries dict
        session_index: Session-to-keys index
    
    Returns:
        True if removed, False if no entries
    """
    if not entries:
        return False
    
    oldest_key = min(
        entries.items(),
        key=lambda item: item[1].get("created_at_epoch", float("inf")),
    )[0]
    
    _, entries = remove_entry(entries, session_index, oldest_key, count_invalidation=False)
    return True
