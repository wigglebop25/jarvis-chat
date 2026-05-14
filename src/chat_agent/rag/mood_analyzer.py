"""Mood learning and correlation analysis for personalized recommendations."""

import logging
from typing import Any

from .detector import MoodDetector
from .cache import MoodCorrelationCache, load_user_actions_from_db, analyze_mood_correlations

logger = logging.getLogger(__name__)


class MoodAnalyzer:
    """Analyzes user query patterns to find mood correlations."""
    
    def __init__(self, cache_ttl_seconds: int = 3600):
        """
        Initialize the mood analyzer.
        
        Args:
            cache_ttl_seconds: TTL for cached correlations (default 1 hour)
        """
        self.detector = MoodDetector()
        self.cache = MoodCorrelationCache(ttl_seconds=cache_ttl_seconds)
    
    def extract_mood_keywords(self, query: str) -> list[str]:
        """
        Extract mood keywords from a user query.
        
        Args:
            query: User query text
            
        Returns:
            List of detected mood keywords
        """
        return self.detector.extract_mood_keywords(query)
    
    def analyze_correlations(self, min_samples: int = 5, top_k: int = 3) -> dict[str, Any]:
        """
        Analyze user action history to find mood-to-entity correlations.
        
        Uses cache to avoid repeated database queries within TTL window.
        
        Args:
            min_samples: Minimum number of samples before accepting a correlation
            top_k: Number of top entities to return per mood
            
        Returns:
            Dictionary mapping mood -> list of correlation objects with confidence
        """
        # Try to use cached result
        cached = self.cache.get_correlations()
        if cached is not None:
            logger.debug("Using cached mood correlations")
            return cached
        
        try:
            # Load actions from database
            actions = load_user_actions_from_db(limit=500)
            
            if not actions:
                logger.debug("No user actions found for mood analysis")
                return {}
            
            # Map queries to moods
            moods_per_action = {}
            for query, tool_name, result_type in actions:
                moods_per_action[query] = self.extract_mood_keywords(query)
            
            # Analyze correlations
            correlations = analyze_mood_correlations(
                moods_per_action=moods_per_action,
                actions=actions,
                min_samples=min_samples,
                top_k=top_k,
            )
            
            # Cache the results
            if correlations:
                self.cache.set_correlations(correlations)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Failed to analyze mood correlations: {e}")
            return {}
    
    def save_mood_correlations(self, correlations: dict[str, Any]) -> int:
        """
        Save analyzed mood correlations back to vector store for RAG retrieval.
        
        Args:
            correlations: Correlations dict from analyze_correlations()
            
        Returns:
            Number of correlations saved
        """
        try:
            from .retriever import get_rag_retriever
            
            retriever = get_rag_retriever()
            saved_count = 0
            
            for mood, entities in correlations.items():
                for entity_info in entities:
                    entity_name = entity_info['entity']
                    confidence = entity_info['confidence']
                    
                    # Create a mood tag in RAG
                    # Note: In a full implementation, we'd link to actual playlist/track IDs
                    if retriever.vector_store.embed_and_cache(
                        entity_id=f"mood:{mood}:{entity_name}",
                        entity_type='mood_tag',
                        text_content=f"Mood '{mood}' correlates with {entity_name}",
                        entity_name=f"{mood} → {entity_name}",
                        metadata={
                            'mood': mood,
                            'entity': entity_name,
                            'confidence': confidence,
                        },
                        ttl_hours=24 * 7,  # Keep for a week
                    ):
                        saved_count += 1
            
            logger.info(f"Saved {saved_count} mood correlations to RAG")
            return saved_count
            
        except Exception as e:
            logger.error(f"Failed to save mood correlations: {e}")
            return 0
    
    def get_recommendations_for_mood(self, mood: str) -> list[str]:
        """
        Get personalized recommendations for a given mood.
        
        Args:
            mood: Mood keyword (e.g., 'sad', 'workout')
            
        Returns:
            List of recommended entities
        """
        try:
            from .retriever import get_rag_retriever
            
            retriever = get_rag_retriever()
            
            # Search RAG for mood correlations
            results = retriever.vector_store.semantic_search(
                f"mood {mood}",
                entity_type='mood_tag',
                top_k=5,
            )
            
            recommendations = []
            for result in results:
                if result['similarity_score'] > 0.5:
                    metadata = result.get('metadata', {})
                    entity = metadata.get('entity')
                    if entity:
                        recommendations.append(entity)
            
            return recommendations[:3]
            
        except Exception as e:
            logger.debug(f"Failed to get mood recommendations: {e}")
            return []
    
    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics about the mood correlation cache.
        
        Returns:
            Dictionary with cache info
        """
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the mood correlation cache."""
        self.cache.invalidate()


# Singleton instance
_mood_analyzer = None


def get_mood_analyzer() -> MoodAnalyzer:
    """Get or create the singleton mood analyzer."""
    global _mood_analyzer
    if _mood_analyzer is None:
        _mood_analyzer = MoodAnalyzer()
    return _mood_analyzer


async def run_mood_correlation_analysis() -> None:
    """
    Background task to periodically update mood correlations.
    
    Call this periodically (e.g., every hour) to keep mood tags fresh.
    """
    try:
        analyzer = get_mood_analyzer()
        
        # Analyze recent user actions
        correlations = analyzer.analyze_correlations(min_samples=3)
        
        # Save findings back to RAG
        if correlations:
            analyzer.save_mood_correlations(correlations)
            logger.info(f"Mood correlation analysis complete: {len(correlations)} moods")
        else:
            logger.debug("No significant mood correlations found in this analysis run")
            
    except Exception as e:
        logger.error(f"Failed to run mood correlation analysis: {e}")
