"""
Intent Matcher

Logic for recognizing intents and extracting parameters.
"""

import re
from typing import Optional

from ..models import Intent, IntentType
from .patterns import INTENT_PATTERNS, PARAMETER_EXTRACTORS


def recognize_intent(text: str) -> Intent:
    """
    Recognize the user's intent from transcribed text.
    """
    text_lower = text.lower().strip()

    if not text_lower:
        return Intent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            raw_text=text
        )

    best_match: Optional[tuple[IntentType, float]] = None

    for intent_type, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                confidence = 0.85

                if len(text_lower.split()) <= 5:
                    confidence += 0.1

                if best_match is None or confidence > best_match[1]:
                    best_match = (intent_type, min(confidence, 1.0))
                break

    if best_match is None:
        return Intent(
            type=IntentType.GENERAL_QUERY,
            confidence=0.6,
            raw_text=text
        )

    intent_type, confidence = best_match
    parameters = extract_parameters(text_lower, intent_type)

    return Intent(
        type=intent_type,
        confidence=confidence,
        parameters=parameters,
        raw_text=text
    )


def extract_parameters(text: str, intent_type: IntentType) -> dict:
    """Extract parameters from text based on intent type."""
    params = {}

    extractors = PARAMETER_EXTRACTORS.get(intent_type, {})

    for param_name, extractor in extractors.items():
        if isinstance(extractor, list):
            for pattern, value in extractor:
                if re.search(pattern, text):
                    params[param_name] = value
                    break
        elif isinstance(extractor, str):
            match = re.search(extractor, text)
            if match:
                # Use entire first group as parameter
                params[param_name] = match.group(1)
        
        # Fallback for simple keyword matches if no specific extractor matched
        if param_name not in params:
             # Specialized logic for resolve_path keywords
             if intent_type == IntentType.PATH_RESOLVE and param_name == "name":
                 for kw in ["downloads", "documents", "desktop", "home", "project"]:
                     if kw in text:
                         params["name"] = kw
                         break

    return params
