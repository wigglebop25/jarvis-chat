"""
agent_logic/mixins.py
──────────────────────
Backward-compatible re-export. Each mixin now lives in its own file:
  - llm_handler.py   → LLMHandlerMixin
  - tool_handler.py  → ToolHandlerMixin
  - cache_handler.py → CacheHandlerMixin
"""
from .llm_handler import LLMHandlerMixin
from .tool_handler import ToolHandlerMixin
from .cache_handler import CacheHandlerMixin

__all__ = ["LLMHandlerMixin", "ToolHandlerMixin", "CacheHandlerMixin"]
