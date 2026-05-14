import json
import logging
from .base import IntentRecognizer
from ..models import Intent, IntentType
from ..llm.base import LLMProvider

logger = logging.getLogger(__name__)

class LLMIntentRecognizer(IntentRecognizer):
    """LLM-based intent recognition."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    async def recognize(self, text: str) -> Intent:
        if not self.llm_provider:
            return Intent(type=IntentType.UNKNOWN, confidence=0.0, raw_text=text)
            
        system_prompt = f"""
        Analyze the user's input and classify it into one of the following intents:
        {", ".join([it.value for it in IntentType])}
        
        Return a JSON object with:
        - "intent": The intent type (lowercase)
        - "confidence": A float between 0 and 1
        - "parameters": A dictionary of extracted parameters (e.g., volume level, folder name)
        
        Input: {text}
        """
        
        try:
            response = await self.llm_provider.complete([{"role": "system", "content": system_prompt}])
            data = json.loads(response.text)
            
            intent_type = IntentType(data.get("intent", "unknown"))
            return Intent(
                type=intent_type,
                confidence=data.get("confidence", 0.0),
                parameters=data.get("parameters", {}),
                raw_text=text
            )
        except Exception as e:
            logger.error(f"LLM intent recognition failed: {e}")
            return Intent(type=IntentType.UNKNOWN, confidence=0.0, raw_text=text)
