"""
context_cache/models.py
────────────────────────
Dataclasses for session context caching:
  CachedMessage         — one conversation turn
  NumericContextArtifact — numeric tensor/array stored in a session
  SessionState          — full session snapshot (messages + artifacts + stats)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CachedMessage:
    role: str
    content: str
    estimated_tokens: int
    created_at_utc: str


@dataclass(slots=True)
class NumericContextArtifact:
    name: str
    values: list[float]
    dtype: str
    updated_at_utc: str


@dataclass(slots=True)
class SessionState:
    session_id: str
    active_dtype: str
    requested_dtype: str
    compatible_dtypes: list[str]
    summary: str = ""
    messages: list[CachedMessage] = field(default_factory=list)
    artifacts: dict[str, NumericContextArtifact] = field(default_factory=dict)
    total_messages_seen: int = 0
    summary_refresh_count: int = 0
    cache_hit_count: int = 0
    estimated_prompt_token_savings: int = 0
    conversion_count: int = 0
    conversion_failures: int = 0
    last_conversion: dict[str, Any] | None = None
