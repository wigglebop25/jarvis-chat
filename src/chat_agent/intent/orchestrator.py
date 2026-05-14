from typing import List
from .base import IntentRecognizer
from ..models import Intent, IntentType

class IntentOrchestrator(IntentRecognizer):
    """Orchestrates multiple intent recognition strategies."""
    
    def __init__(self, recognizers: List[IntentRecognizer]):
        self.recognizers = recognizers

    async def recognize(self, text: str) -> Intent:
        best_intent = Intent(type=IntentType.UNKNOWN, confidence=0.0, raw_text=text)
        
        for recognizer in self.recognizers:
            intent = await recognizer.recognize(text)
            if intent.confidence > 0.8: # Threshold for "good enough"
                return intent
            if intent.confidence > best_intent.confidence:
                best_intent = intent
                
        return best_intent
