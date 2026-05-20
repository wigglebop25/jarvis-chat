"""
rag/llm_integration.py
────────────────────────
Backward-compatible re-export. Implementation split into:
  - rag/augmentation.py   — augment_messages_with_rag
  - rag/result_caching.py — cache_tool_results
"""
from .augmentation import augment_messages_with_rag, augment_messages_with_rag_async
from .result_caching import cache_tool_results

__all__ = [
    "augment_messages_with_rag",
    "augment_messages_with_rag_async",
    "cache_tool_results",
]
