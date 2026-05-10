"""Mood learning and correlation analysis for personalized recommendations."""

import logging
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".jarvis" / "cache"
VECTOR_DB_PATH = CACHE_DIR / "vector_store.db"


class MoodAnalyzer:
    """Analyzes user query patterns to find mood correlations."""
    
    # Mood keywords and their related terms
    MOOD_KEYWORDS = {
        'happy': ['happy', 'cheerful', 'upbeat', 'energetic', 'fun', 'positive', 'good', 'awesome'],
        'sad': ['sad', 'sad', 'cry', 'crying', 'blue', 'down', 'depressed', 'heartbreak', 'lonely'],
        'workout': ['workout', 'gym', 'exercise', 'run', 'running', 'train', 'pump', 'pump up', 'energy'],
        'chill': ['chill', 'relax', 'calm', 'peace', 'quiet', 'lounge', 'mellow', 'laid back', 'cool'],
        'focused': ['focus', 'study', 'work', 'coding', 'productive', 'concentrate', 'thinking', 'thinking'],
        'party': ['party', 'dance', 'club', 'dance', 'dancing', 'celebrate', 'celebration', 'fun'],
        'romantic': ['love', 'romantic', 'date', 'romance', 'sweet', 'intimate', 'cozy'],
        'sleep': ['sleep', 'sleepy', 'tired', 'bed', 'night', 'before sleep', 'relaxing'],
    }
    
    def __init__(self):
        """Initialize the mood analyzer."""
        self.mood_regex = self._compile_mood_regex()
    
    def _compile_mood_regex(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for each mood."""
        patterns = {}
        for mood, keywords in self.MOOD_KEYWORDS.items():
            # Create case-insensitive regex with word boundaries
            pattern_str = r'\b(' + '|'.join(keywords) + r')\b'
            patterns[mood] = re.compile(pattern_str, re.IGNORECASE)
        return patterns
    
    def extract_mood_keywords(self, query: str) -> list[str]:
        """
        Extract mood keywords from a user query.
        
        Args:
            query: User query text
            
        Returns:
            List of detected mood keywords
        """
        moods = []
        for mood, pattern in self.mood_regex.items():
            if pattern.search(query):
                moods.append(mood)
        return moods
    
    def analyze_correlations(self, min_samples: int = 5, top_k: int = 3) -> dict[str, Any]:
        """
        Analyze user action history to find mood-to-entity correlations.
        
        Args:
            min_samples: Minimum number of samples before accepting a correlation
            top_k: Number of top entities to return per mood
            
        Returns:
            Dictionary mapping mood -> list of (entity_id, entity_name, confidence)
        """
        try:
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()
            
            # Get all user actions
            cursor.execute("""
                SELECT query, tool_name, result_type FROM user_actions
                ORDER BY timestamp DESC
                LIMIT 500
            """)
            
            actions = cursor.fetchall()
            conn.close()
            
            if not actions:
                logger.debug("No user actions found for mood analysis")
                return {}
            
            # Group by detected mood
            mood_correlations = {}
            
            for query, tool_name, result_type in actions:
                moods = self.extract_mood_keywords(query)
                
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
