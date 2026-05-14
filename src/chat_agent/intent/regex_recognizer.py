import re
import logging
from typing import Optional
from .base import IntentRecognizer
from .patterns import INTENT_PATTERNS, PARAMETER_EXTRACTORS
from ..models import Intent, IntentType

logger = logging.getLogger(__name__)


class RegexIntentRecognizer(IntentRecognizer):
    """Regex-based intent recognition."""
    
    # Class variables for pre-compiled patterns
    _compiled_intent_patterns: dict[IntentType, list[re.Pattern]] = {}
    _compiled_parameter_patterns: dict[IntentType, dict[str, list[tuple[re.Pattern, str]] | re.Pattern]] = {}
    _patterns_compiled = False
    
    @classmethod
    def _compile_patterns(cls) -> None:
        """Pre-compile all regex patterns on first use."""
        if cls._patterns_compiled:
            return
        
        logger.debug("Pre-compiling regex patterns for intent recognition")
        
        # Compile intent patterns
        for intent_type, patterns in INTENT_PATTERNS.items():
            cls._compiled_intent_patterns[intent_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Compile parameter extractor patterns
        for intent_type, extractors in PARAMETER_EXTRACTORS.items():
            cls._compiled_parameter_patterns[intent_type] = {}
            
            for param_name, extractor in extractors.items():
                if isinstance(extractor, list):
                    # List of (pattern, value) tuples
                    compiled_list = [
                        (re.compile(pattern, re.IGNORECASE), value)
                        for pattern, value in extractor
                    ]
                    cls._compiled_parameter_patterns[intent_type][param_name] = compiled_list
                elif isinstance(extractor, str):
                    # Single pattern string
                    cls._compiled_parameter_patterns[intent_type][param_name] = re.compile(extractor, re.IGNORECASE)
        
        cls._patterns_compiled = True
    
    async def recognize(self, text: str) -> Intent:
        text_lower = text.lower().strip()
        if not text_lower:
            return Intent(type=IntentType.UNKNOWN, confidence=0.0, raw_text=text)

        # Ensure patterns are compiled
        self._compile_patterns()

        best_match: Optional[tuple[IntentType, float]] = None

        for intent_type, compiled_patterns in self._compiled_intent_patterns.items():
            for compiled_pattern in compiled_patterns:
                if compiled_pattern.search(text_lower):
                    confidence = 0.85
                    if len(text_lower.split()) <= 5:
                        confidence += 0.1
                    if best_match is None or confidence > best_match[1]:
                        best_match = (intent_type, min(confidence, 1.0))
                    break

        if best_match is None:
            return Intent(type=IntentType.UNKNOWN, confidence=0.0, raw_text=text)

        intent_type, confidence = best_match
        parameters = self.extract_parameters(text_lower, intent_type)

        return Intent(
            type=intent_type,
            confidence=confidence,
            parameters=parameters,
            raw_text=text
        )

    def extract_parameters(self, text: str, intent_type: IntentType) -> dict:
        """Extract parameters from text based on intent type."""
        # Ensure patterns are compiled
        self._compile_patterns()
        
        params = {}
        compiled_extractors = self._compiled_parameter_patterns.get(intent_type, {})

        for param_name, extractor in compiled_extractors.items():
            if isinstance(extractor, list):
                # List of (compiled_pattern, value) tuples
                for compiled_pattern, value in extractor:
                    if compiled_pattern.search(text):
                        params[param_name] = value
                        break
            elif isinstance(extractor, re.Pattern):
                # Single compiled pattern
                match = extractor.search(text)
                if match:
                    params[param_name] = match.group(1)
            
            if param_name not in params:
                 if intent_type == IntentType.PATH_RESOLVE and param_name == "name":
                     for kw in ["downloads", "documents", "desktop", "home", "project"]:
                         if kw in text:
                             params["name"] = kw
                             break
        return params
