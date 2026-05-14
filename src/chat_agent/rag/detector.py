"""Text analysis logic for mood detection and sentiment scoring."""

import logging
import re

logger = logging.getLogger(__name__)


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


class MoodDetector:
    """Detects mood keywords in text using regex patterns."""
    
    def __init__(self):
        """Initialize mood detector with compiled patterns."""
        self.mood_regex = self._compile_mood_regex()
    
    def _compile_mood_regex(self) -> dict[str, re.Pattern]:
        """
        Compile regex patterns for each mood.
        
        Returns:
            Dictionary mapping mood -> compiled regex pattern
        """
        patterns = {}
        for mood, keywords in MOOD_KEYWORDS.items():
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
            List of detected mood keywords (e.g., ['happy', 'energetic'])
        """
        moods = []
        for mood, pattern in self.mood_regex.items():
            if pattern.search(query):
                moods.append(mood)
        return moods
    
    def score_sentiment(self, query: str, detected_moods: list[str]) -> float:
        """
        Calculate sentiment score (0.0 to 1.0) based on detected moods.
        
        Args:
            query: Original query text
            detected_moods: List of moods detected in query
            
        Returns:
            Float between 0.0 (negative) and 1.0 (positive)
        """
        if not detected_moods:
            return 0.5  # Neutral
        
        # Map moods to sentiment scores
        sentiment_map = {
            'happy': 0.9,
            'sad': 0.2,
            'workout': 0.8,
            'chill': 0.7,
            'focused': 0.75,
            'party': 0.85,
            'romantic': 0.8,
            'sleep': 0.6,
        }
        
        scores = [sentiment_map.get(mood, 0.5) for mood in detected_moods]
        return sum(scores) / len(scores) if scores else 0.5
