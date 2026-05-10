"""Test data fixtures and helper utilities for RAG tests."""

import time
from typing import Any


class TestDataFixtures:
    """Reusable test data for all RAG tests."""
    
    PLAYLISTS = [
        ("playlist:test:1", "Sad Indie Mix", "For sad indie lovers"),
        ("playlist:test:2", "Chill Vibes", "Relaxing background music"),
        ("playlist:test:3", "High Energy Beats", "Workout motivation"),
    ]
    
    QUERIES = {
        "exact_match": "sad indie",
        "related_term": "crying songs",
        "irrelevant": "quantum physics",
    }
    
    MOOD_QUERIES = [
        ("play sad songs", "sad"),
        ("I need workout music", "workout"),
        ("relaxation time", "chill"),
        ("party mode", "party"),
    ]
    
    @staticmethod
    async def setup_test_store():
        """Create a test vector store with sample data."""
        from ..vector_store import get_vector_store
        
        store = get_vector_store(":memory:")
        for pl_id, pl_name, description in TestDataFixtures.PLAYLISTS:
            try:
                store.embed_and_cache(pl_id, "playlist", description, entity_name=pl_name, ttl_hours=24)
            except Exception:
                pass
        
        return store
    
    @staticmethod
    async def setup_mood_tracker():
        """Create mood tracker with sample actions."""
        from ..mood_integration import MoodTracker
        
        tracker = MoodTracker()
        for query, mood in TestDataFixtures.MOOD_QUERIES:
            tracker.log_action(query, "playMusic", mood)
        
        return tracker


class LatencyBenchmark:
    """Helper for latency measurements and assertions."""
    
    def __init__(self, name: str):
        self.name = name
        self.measurements = []
    
    async def measure(self, func, *args, **kwargs) -> Any:
        """Measure function latency."""
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        self.measurements.append(elapsed)
        return result
    
    def avg_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self.measurements:
            return 0
        return sum(self.measurements) / len(self.measurements)
    
    def max_ms(self) -> float:
        """Maximum latency in milliseconds."""
        return max(self.measurements) if self.measurements else 0
    
    def min_ms(self) -> float:
        """Minimum latency in milliseconds."""
        return min(self.measurements) if self.measurements else 0
    
    def report(self) -> str:
        """Generate benchmark report."""
        if not self.measurements:
            return f"{self.name}: No measurements"
        return (
            f"{self.name}: avg={self.avg_ms():.2f}ms, "
            f"min={self.min_ms():.2f}ms, max={self.max_ms():.2f}ms"
        )


class TestAssertions:
    """Custom assertions for RAG tests."""
    
    @staticmethod
    def assert_similarity_in_range(similarity: float, min_val: float, max_val: float):
        """Assert similarity score is within expected range."""
        assert min_val <= similarity <= max_val, (
            f"Similarity {similarity} not in range [{min_val}, {max_val}]"
        )
    
    @staticmethod
    def assert_results_found(results: list, min_count: int = 1):
        """Assert retrieval found expected results."""
        assert len(results) >= min_count, (
            f"Expected at least {min_count} results, got {len(results)}"
        )
    
    @staticmethod
    def assert_latency_acceptable(elapsed_ms: float, max_ms: float):
        """Assert latency is below threshold."""
        assert elapsed_ms <= max_ms, (
            f"Latency {elapsed_ms:.2f}ms exceeds limit {max_ms}ms"
        )
    
    @staticmethod
    def assert_cache_valid(item: dict, key: str):
        """Assert cached item has required keys."""
        assert key in item, f"Cache item missing required key: {key}"
