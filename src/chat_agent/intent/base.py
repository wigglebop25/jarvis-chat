from typing import Protocol, runtime_checkable
from ..models import Intent

@runtime_checkable
class IntentRecognizer(Protocol):
    """Protocol for intent recognition strategies."""
    
    async def recognize(self, text: str) -> Intent:
        """Recognize intent from text."""
        ...
