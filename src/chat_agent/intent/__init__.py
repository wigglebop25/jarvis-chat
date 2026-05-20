"""
Intent Recognition Module

Recognize user intents from transcribed text.
Maps voice commands to actionable intents.
"""

from .matcher import recognize_intent, extract_parameters
from .mappings import get_tool_name_for_intent, map_intent_params_to_tool

__all__ = [
    "recognize_intent",
    "extract_parameters",
    "get_tool_name_for_intent",
    "map_intent_params_to_tool",
]
