"""Mood analysis tests for RAG system."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def test_mood_analysis(mood_analyzer: Any) -> dict:
    """
    Test mood extraction and correlation analysis.
    
    Returns:
        Dict with test results
    """
    results = {
        "extraction": False,
        "correlation": False,
        "confidence": 0.0,
        "errors": [],
    }
    
    try:
        logger.info("\n[TEST 4] Mood Analysis...")
        
        # Test 1: Mood keyword extraction
        test_queries = [
            ("play sad songs", "sad"),
            ("I need workout music", "workout"),
            ("relaxation time", "chill"),
        ]
        
        from ..mood_analyzer import get_mood_analyzer
        analyzer = get_mood_analyzer()
        
        for query, expected_mood in test_queries:
            moods = analyzer.extract_mood_keywords(query)
            if expected_mood in moods or any(m in query.lower() for m in moods):
                logger.info(f"✓ Mood extraction: '{query}' → {moods}")
                results["extraction"] = True
            else:
                logger.warning(f"✗ Mood extraction: '{query}' missing {expected_mood}")
        
        # Test 2: Correlation analysis
        if hasattr(mood_analyzer, 'analyze_correlations'):
            correlations = mood_analyzer.analyze_correlations(min_samples=1)
            
            if correlations:
                logger.info(f"✓ Correlations found: {len(correlations)} moods")
                for mood, entities in correlations.items():
                    logger.info(f"  - {mood}: {len(entities)} entities")
                
                results["correlation"] = True
                
                # Calculate average confidence
                total_confidence = 0
                total_entities = 0
                for mood_data in correlations.values():
                    for entity in mood_data:
                        total_confidence += entity.get("confidence", 0)
                        total_entities += 1
                
                if total_entities > 0:
                    results["confidence"] = total_confidence / total_entities
                    logger.info(f"✓ Average confidence: {results['confidence']:.0%}")
            else:
                logger.info("ℹ No correlations found (may need more samples)")
        
    except Exception as e:
        logger.error(f"Mood analysis test failed: {e}")
        results["errors"].append(str(e))
    
    return results
