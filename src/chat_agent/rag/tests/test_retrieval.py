"""Retrieval quality tests for RAG semantic search."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def test_retrieval_quality(vector_store: Any, embeddings: dict[str, Any]) -> dict:
    """
    Test retrieval quality with semantic search.
    
    Returns:
        Dict with test results
    """
    results: dict[str, Any] = {
        "exact_match": False,
        "related_term": False,
        "irrelevant_filter": False,
        "errors": [],
    }
    
    try:
        # Test 1: Exact match retrieval
        logger.info("[TEST 1] Retrieval Quality...")
        search_results = vector_store.semantic_search("sad indie", "playlist", top_k=3)
        
        if search_results and search_results[0]["similarity"] > 0.7:
            logger.info("✓ Exact match: 'sad indie' → found with score > 0.7")
            results["exact_match"] = True
        else:
            logger.warning("✗ Exact match: expected score > 0.7")
        
        # Test 2: Related term retrieval
        search_results = vector_store.semantic_search("crying songs", "playlist", top_k=3)
        
        if search_results and search_results[0]["similarity"] > 0.5:
            logger.info("✓ Related term: 'crying songs' → found with score > 0.5")
            results["related_term"] = True
        else:
            logger.warning("✗ Related term: expected score > 0.5")
        
        # Test 3: Irrelevant filter
        search_results = vector_store.semantic_search("quantum physics", "playlist", top_k=3)
        
        if not search_results or search_results[0]["similarity"] < 0.3:
            logger.info("✓ Irrelevant filter: 'quantum physics' → correctly filtered")
            results["irrelevant_filter"] = True
        else:
            logger.warning("✗ Irrelevant filter: expected low similarity")
        
    except Exception as e:
        logger.error(f"Retrieval quality test failed: {e}")
        results["errors"].append(str(e))
    
    return results
