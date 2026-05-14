"""Cache and TTL eviction tests for RAG system."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def test_ttl_eviction(vector_store: Any) -> dict:
    """
    Test TTL eviction and cache freshness.
    
    Returns:
        Dict with test results
    """
    results: dict[str, Any] = {
        "expiration": False,
        "retention": False,
        "cleanup": False,
        "errors": [],
    }
    
    try:
        logger.info("\n[TEST 2] TTL Eviction & Cache Freshness...")
        
        # Test 1: Item expires and is removed
        logger.info("Creating test item with 2-second TTL...")
        test_id = "test:ttl:1"
        vector_store.embed_and_cache(
            test_id,
            "test entity",
            "test content",
            ttl_hours=0.00055,  # ~2 seconds
        )
        
        # Verify item exists
        items = vector_store.get_by_type("test")
        if items:
            logger.info(f"✓ Item cached: {len(items)} items in store")
            results["retention"] = True
        
        # Wait for expiration
        logger.info("Waiting 3 seconds for TTL to expire...")
        await asyncio.sleep(3)
        
        # Verify expired item is removed
        deleted = vector_store.clear_stale()
        if deleted > 0:
            logger.info(f"✓ TTL expiration: {deleted} expired items removed")
            results["expiration"] = True
        
        # Verify store is now empty
        items_after = vector_store.get_by_type("test")
        if not items_after:
            logger.info("✓ Store cleanup: expired items removed")
            results["cleanup"] = True
        else:
            logger.warning(f"✗ Store cleanup: {len(items_after)} items still present")
        
    except Exception as e:
        logger.error(f"TTL eviction test failed: {e}")
        results["errors"].append(str(e))
    
    return results


async def test_cache_consistency(cache_store: Any) -> dict:
    """Test tool cache consistency and hit rates."""
    results: dict[str, Any] = {
        "store": False,
        "retrieve": False,
        "hit_rate": 0.0,
        "errors": [],
    }
    
    try:
        logger.info("\nTesting cache consistency...")
        
        # Store item
        cache_key = "test:cache:1"
        test_data = {"tool": "test", "result": "cached"}
        
        cache_store.set(cache_key, test_data, ttl_seconds=3600)
        results["store"] = True
        logger.info("✓ Cache store successful")
        
        # Retrieve item
        retrieved = cache_store.get(cache_key)
        if retrieved == test_data:
            results["retrieve"] = True
            logger.info("✓ Cache retrieval successful")
        
        # Calculate hit rate
        stats = cache_store.get_stats()
        if stats:
            results["hit_rate"] = stats.get("hit_rate", 0.0)
            logger.info(f"✓ Cache hit rate: {results['hit_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Cache consistency test failed: {e}")
        results["errors"].append(str(e))
    
    return results
