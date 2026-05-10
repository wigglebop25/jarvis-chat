"""Latency benchmark tests for RAG components."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def test_latency_benchmarks(vector_store: Any, embeddings: dict[str, Any]) -> dict:
    """
    Benchmark RAG component latencies.
    
    Returns:
        Dict with latency measurements
    """
    results = {
        "embedding_latency_ms": 0.0,
        "search_latency_ms": 0.0,
        "cache_store_latency_ms": 0.0,
        "cache_hit_latency_ms": 0.0,
        "errors": [],
    }
    
    try:
        logger.info("\n[TEST 3] Latency Benchmarks...")
        
        # Benchmark 1: Embedding latency
        from ..embedding_model import embed_text
        
        start = time.perf_counter()
        embed_text("test query")
        elapsed = (time.perf_counter() - start) * 1000
        
        results["embedding_latency_ms"] = elapsed
        logger.info(f"✓ Embedding latency: {elapsed:.2f}ms")
        
        # Benchmark 2: Vector search latency
        start = time.perf_counter()
        vector_store.semantic_search("test", "playlist", top_k=5)
        elapsed = (time.perf_counter() - start) * 1000
        
        results["search_latency_ms"] = elapsed
        logger.info(f"✓ Vector search latency: {elapsed:.2f}ms")
        
        # Benchmark 3: Tool cache store latency
        from ...tools.tool_cache import set_tool_cache
        
        start = time.perf_counter()
        set_tool_cache("test_tool", {"result": "data"})
        elapsed = (time.perf_counter() - start) * 1000
        
        results["cache_store_latency_ms"] = elapsed
        logger.info(f"✓ Cache store latency: {elapsed:.2f}ms")
        
        # Benchmark 4: Tool cache hit latency
        from ...tools.tool_cache import get_tool_cache
        
        start = time.perf_counter()
        get_tool_cache("test_tool")
        elapsed = (time.perf_counter() - start) * 1000
        
        results["cache_hit_latency_ms"] = elapsed
        logger.info(f"✓ Cache hit latency: {elapsed:.2f}ms")
        
    except Exception as e:
        logger.error(f"Latency benchmark failed: {e}")
        results["errors"].append(str(e))
    
    return results
