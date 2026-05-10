"""Phase 4: Mood learning integration - tracks queries and learns user patterns."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MoodTracker:
    """Class wrapper for mood tracking operations."""
    
    def log_action(self, query: str, tool_name: str, result_type: str):
        """Log action for mood analysis (sync wrapper)."""
        import asyncio
        try:
            # Create a simple loop-runner for the async tracking
            asyncio.run(track_user_action(query, tool_name, result_type))
        except Exception:
            pass


async def track_user_action(
    query: str,
    tool_name: Optional[str] = None,
    tool_result: Optional[Any] = None,
) -> None:
    """
    Log a user action for mood correlation analysis.
    
    Args:
        query: Original user query
        tool_name: Tool that was executed
        tool_result: Result from the tool
    """
    try:
        from ..rag import get_rag_retriever, get_mood_analyzer
        
        # Extract keywords from result
        result_type = type(tool_result).__name__ if tool_result else None
        
        # Log to vector store
        retriever = get_rag_retriever()
        retriever.vector_store.log_user_action(
            query=query,
            tool_name=tool_name,
            result_type=result_type,
        )
        
        # Also analyze for mood keywords
        analyzer = get_mood_analyzer()
        moods = analyzer.extract_mood_keywords(query)
        
        if moods:
            logger.debug(f"Detected moods in query: {moods}")
            # Moods will be included in future mood correlation analysis
        
    except Exception as e:
        logger.debug(f"Failed to track user action: {e}")


async def cache_tool_result_for_rag(
    tool_name: str,
    tool_result: Any,
) -> None:
    """
    Cache tool results in RAG for semantic search.
    
    Args:
        tool_name: Tool that executed
        tool_result: Result from the tool
    """
    try:
        from ..rag.llm_integration import cache_tool_results
        
        # Use the existing caching function
        cache_tool_results(tool_name, tool_result)
        
    except Exception as e:
        logger.debug(f"Failed to cache tool result: {e}")


async def periodically_update_mood_correlations(force: bool = False) -> None:
    """
    Periodically analyze user actions and update mood correlations.
    
    Args:
        force: If True, run analysis immediately; otherwise only run if enough actions logged
    """
    try:
        from ..rag import get_mood_analyzer
        
        analyzer = get_mood_analyzer()
        
        # Analyze mood correlations (minimum 3 samples per mood)
        correlations = analyzer.analyze_correlations(min_samples=3, top_k=3)
        
        if correlations:
            # Save to RAG for future retrieval
            saved_count = analyzer.save_mood_correlations(correlations)
            logger.info(f"Mood analysis complete: {saved_count} mood tags saved")
        
    except Exception as e:
        logger.debug(f"Failed to update mood correlations: {e}")
