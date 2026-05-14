"""Main RAG test suite runner."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RAGTestSuite:
    """Orchestrate all RAG verification tests."""
    
    def __init__(self):
        """Initialize test suite."""
        self.results = {}
    
    async def run_all_tests(self, verbose: bool = True) -> dict:
        """Run all RAG tests."""
        from .test_retrieval import test_retrieval_quality
        from .test_cache import test_ttl_eviction
        from .test_latency import test_latency_benchmarks
        from .test_mood import test_mood_analysis
        from ..vector_store import get_vector_store
        from ..mood_analyzer import get_mood_analyzer
        
        logger.info("=" * 80)
        logger.info("PHASE 5: RAG VERIFICATION TEST SUITE")
        logger.info("=" * 80)
        logger.info("")
        
        try:
            # Setup
            vector_store = get_vector_store()
            mood_analyzer = get_mood_analyzer()
            embeddings: dict[str, Any] = {}
            
            # Run tests
            self.results["retrieval_quality"] = await test_retrieval_quality(
                vector_store, embeddings
            )
            
            self.results["ttl_eviction"] = await test_ttl_eviction(vector_store)
            
            self.results["latency"] = await test_latency_benchmarks(
                vector_store, embeddings
            )
            
            self.results["mood_analysis"] = await test_mood_analysis(mood_analyzer)
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.results["error"] = str(e)
        
        return self.results
    
    def _print_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        
        total_passed = 0
        total_failed = 0
        
        for test_name, results in self.results.items():
            if isinstance(results, dict):
                passed = sum(1 for k, v in results.items() 
                           if k != "errors" and v is True)
                failed = sum(1 for k, v in results.items() 
                           if k != "errors" and v is False)
                
                total_passed += passed
                total_failed += failed
                
                status = "✓ PASS" if failed == 0 else "✗ FAIL"
                logger.info(f"{status} | {test_name}: {passed} passed, {failed} failed")
        
        logger.info("\n" + "=" * 80)
        if total_failed == 0:
            logger.info(f"✓ ALL TESTS PASSED ({total_passed} assertions)")
        else:
            logger.info(f"✗ SOME TESTS FAILED ({total_passed} passed, {total_failed} failed)")
        logger.info("=" * 80)


async def run_rag_verification_tests() -> dict:
    """Run RAG verification tests and return results."""
    suite = RAGTestSuite()
    return await suite.run_all_tests()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_rag_verification_tests())
