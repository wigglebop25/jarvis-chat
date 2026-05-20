"""
context_cache.py
─────────────────
Backward-compatible re-export. Implementation split into:
  - context_cache/models.py       — CachedMessage, NumericContextArtifact, SessionState
  - context_cache/session_cache.py — SessionContextCache
"""
from .context_cache.models import CachedMessage, NumericContextArtifact, SessionState
from .context_cache.session_cache import SessionContextCache

__all__ = [
    "CachedMessage",
    "NumericContextArtifact",
    "SessionState",
    "SessionContextCache",
]
