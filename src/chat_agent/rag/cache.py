"""Emotion cache with TTL and invalidation support."""

import logging
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".jarvis" / "cache"
VECTOR_DB_PATH = CACHE_DIR / "vector_store.db"


class MoodCorrelationCache:
    """Manages mood correlation cache with TTL and statistics."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize mood correlation cache.
        
        Args:
            ttl_seconds: Time-to-live for cached correlations (default 1 hour)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_analysis_time: Optional[float] = None
    
    def get_correlations(self) -> Optional[dict[str, Any]]:
        """
        Get cached correlations if still valid.
        
        Returns:
            Cached correlations dict, or None if cache expired
        """
        if not self._cache:
            return None
        
        if self._last_analysis_time is None:
            return None
        
        # Check if cache has expired
        if time.time() - self._last_analysis_time > self.ttl_seconds:
            logger.debug("Mood correlation cache expired, clearing")
            self.invalidate()
            return None
        
        return self._cache.get("correlations")
    
    def set_correlations(self, correlations: dict[str, Any]) -> None:
        """
        Cache analyzed correlations.
        
        Args:
            correlations: Dictionary mapping mood -> list of correlation objects
        """
        self._cache["correlations"] = correlations
        self._last_analysis_time = time.time()
        logger.debug(f"Cached {len(correlations)} mood correlations")
    
    def invalidate(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._last_analysis_time = None
        logger.debug("Mood correlation cache invalidated")
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache info (size, age, ttl)
        """
        if not self._cache or self._last_analysis_time is None:
            return {"size": 0, "valid": False, "age_seconds": 0}
        
        correlations = self._cache.get("correlations", {})
        age = time.time() - self._last_analysis_time
        
        return {
            "size": len(correlations),
            "valid": age < self.ttl_seconds,
            "age_seconds": age,
            "ttl_seconds": self.ttl_seconds,
        }


def load_user_actions_from_db(limit: int = 500) -> list[tuple[str, str, str]]:
    """
    Load user actions from vector store database.
    
    Args:
        limit: Maximum number of actions to retrieve
        
    Returns:
        List of (query, tool_name, result_type) tuples
    """
    try:
        conn = sqlite3.connect(str(VECTOR_DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query, tool_name, result_type FROM user_actions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        actions = cursor.fetchall()
        conn.close()
        
        logger.debug(f"Loaded {len(actions)} user actions from database")
        return actions
    
    except Exception as e:
        logger.error(f"Failed to load user actions: {e}")
        return []


def analyze_mood_correlations(
    moods_per_action: dict[str, list[str]],
    actions: list[tuple[str, str, str]],
    min_samples: int = 5,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Analyze correlations between moods and entities from user actions.
    
    Args:
        moods_per_action: Dictionary mapping query -> list of detected moods
        actions: List of (query, tool_name, result_type) tuples
        min_samples: Minimum samples before including a mood
        top_k: Number of top entities to return per mood
        
    Returns:
        Dictionary mapping mood -> list of correlation objects with confidence
    """
    if not actions:
        logger.debug("No user actions provided for mood correlation analysis")
        return {}
    
    # Group by detected mood
    mood_correlations: dict[str, Counter] = {}
    
    for query, tool_name, result_type in actions:
        moods = moods_per_action.get(query, [])
        
        for mood in moods:
            if mood not in mood_correlations:
                mood_correlations[mood] = Counter()
            
            # Count correlations (preference: result_type > tool_name)
            key = result_type or tool_name or 'unknown'
            mood_correlations[mood][key] += 1
    
    # Filter and format results
    result = {}
    for mood, correlations in mood_correlations.items():
        total = sum(correlations.values())
        
        # Only include moods with enough samples
        if total < min_samples:
            continue
        
        # Get top-k with confidence scores
        top_items = []
        for entity, count in correlations.most_common(top_k):
            confidence = count / total
            top_items.append({
                'entity': entity,
                'count': count,
                'confidence': confidence,
            })
        
        if top_items:
            result[mood] = top_items
    
    logger.info(f"Found {len(result)} mood correlations from {len(actions)} actions")
    return result
