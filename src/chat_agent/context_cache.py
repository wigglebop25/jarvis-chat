"""
Session context cache for conversation persistence and token budgeting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .context_dtype import (
    DTypeConversionResult,
    compatible_dtypes_for_model,
    convert_dtype,
    resolve_target_dtype,
    validate_context_dtype,
)


def estimate_text_tokens(text: str) -> int:
    """
    Lightweight token estimate used for context budgeting.

    This is intentionally simple and deterministic for consistent comparisons.
    """
    if not text.strip():
        return 0
    return max(1, int(len(text) / 4))


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


class SessionContextCache:
    """Session-scoped cache with summary compression and dtype-aware artifacts."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        requested_dtype: str = "fp16",
        max_turns: int = 20,
        summary_keep_last: int = 8,
        token_budget: int = 3000,
        persistence_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_turns = max_turns
        self.summary_keep_last = summary_keep_last
        self.token_budget = token_budget
        self.persistence_path = persistence_path
        self._sessions: dict[str, SessionState] = {}

        compatible = compatible_dtypes_for_model(provider=provider, model=model)
        self.compatible_dtypes = sorted(compatible)
        self.requested_dtype = validate_context_dtype(requested_dtype)
        self.default_dtype = resolve_target_dtype(self.requested_dtype, compatible)

        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _summarize_messages(self, messages: list[CachedMessage]) -> str:
        lines: list[str] = []
        for msg in messages:
            words = msg.content.split()
            excerpt = " ".join(words[:24]).strip()
            if len(words) > 24:
                excerpt += " ..."
            lines.append(f"- {msg.role}: {excerpt}")
        return "\n".join(lines)

    def _get_or_create(self, session_id: str) -> SessionState:
        if session_id in self._sessions:
            return self._sessions[session_id]

        state = SessionState(
            session_id=session_id,
            active_dtype=self.default_dtype,
            requested_dtype=self.requested_dtype,
            compatible_dtypes=self.compatible_dtypes,
        )
        self._sessions[session_id] = state
        return state

    def _maybe_compact(self, state: SessionState) -> None:
        if len(state.messages) <= self.max_turns:
            return

        compact_candidates = state.messages[: -self.summary_keep_last]
        compacted = self._summarize_messages(compact_candidates)
        if state.summary:
            state.summary = f"{state.summary}\n{compacted}".strip()
        else:
            state.summary = compacted
        state.messages = state.messages[-self.summary_keep_last :]
        state.summary_refresh_count += 1

    def add_message(self, session_id: str, role: str, content: str) -> None:
        state = self._get_or_create(session_id)
        state.messages.append(
            CachedMessage(
                role=role,
                content=content,
                estimated_tokens=estimate_text_tokens(content),
                created_at_utc=self._now(),
            )
        )
        state.total_messages_seen += 1
        self._maybe_compact(state)
        self._persist()

    def build_messages(self, session_id: str, system_prompt: str) -> list[dict[str, str]]:
        state = self._get_or_create(session_id)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        used_tokens = estimate_text_tokens(system_prompt)

        if state.summary:
            summary_message = f"Conversation summary:\n{state.summary}"
            summary_tokens = estimate_text_tokens(summary_message)
            if used_tokens + summary_tokens <= self.token_budget:
                messages.append({"role": "system", "content": summary_message})
                used_tokens += summary_tokens

        total_message_tokens = sum(m.estimated_tokens for m in state.messages)
        selected: list[CachedMessage] = []
        selected_tokens = 0

        for msg in reversed(state.messages):
            if used_tokens + selected_tokens + msg.estimated_tokens > self.token_budget:
                continue
            selected.append(msg)
            selected_tokens += msg.estimated_tokens

        selected.reverse()
        dropped_tokens = max(0, total_message_tokens - selected_tokens)
        if dropped_tokens > 0:
            state.cache_hit_count += 1
            state.estimated_prompt_token_savings += dropped_tokens

        for msg in selected:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def register_artifact(
        self,
        session_id: str,
        name: str,
        values: list[float],
        source_dtype: str = "fp32",
    ) -> DTypeConversionResult:
        state = self._get_or_create(session_id)
        result = convert_dtype(values, source_dtype=source_dtype, target_dtype=state.active_dtype)
        state.artifacts[name] = NumericContextArtifact(
            name=name,
            values=result.values,
            dtype=result.target_dtype,
            updated_at_utc=self._now(),
        )
        state.conversion_count += 1
        state.last_conversion = result.to_dict()
        self._persist()
        return result

    def convert_artifact_dtype(
        self,
        session_id: str,
        name: str,
        target_dtype: str,
    ) -> DTypeConversionResult:
        state = self._get_or_create(session_id)
        target = validate_context_dtype(target_dtype)
        if target not in state.compatible_dtypes:
            state.conversion_failures += 1
            raise ValueError(
                f"dtype '{target}' not compatible with provider/model profile. "
                f"Compatible: {state.compatible_dtypes}"
            )

        artifact = state.artifacts.get(name)
        if not artifact:
            raise KeyError(f"artifact '{name}' not found for session '{session_id}'")

        result = convert_dtype(
            values=artifact.values,
            source_dtype=artifact.dtype,
            target_dtype=target,
        )
        artifact.values = result.values
        artifact.dtype = result.target_dtype
        artifact.updated_at_utc = self._now()
        state.active_dtype = target
        state.conversion_count += 1
        state.last_conversion = result.to_dict()
        self._persist()
        return result

    def get_stats(self, session_id: str) -> dict[str, Any]:
        state = self._get_or_create(session_id)
        return {
            "session_id": state.session_id,
            "requested_dtype": state.requested_dtype,
            "active_dtype": state.active_dtype,
            "compatible_dtypes": state.compatible_dtypes,
            "message_count": len(state.messages),
            "summary_present": bool(state.summary),
            "summary_refresh_count": state.summary_refresh_count,
            "cache_hit_count": state.cache_hit_count,
            "estimated_prompt_token_savings": state.estimated_prompt_token_savings,
            "conversion_count": state.conversion_count,
            "conversion_failures": state.conversion_failures,
            "last_conversion": state.last_conversion,
        }

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._persist()

    def _persist(self) -> None:
        if not self.persistence_path:
            return

        payload: dict[str, Any] = {}
        for session_id, state in self._sessions.items():
            payload[session_id] = {
                "session_id": state.session_id,
                "active_dtype": state.active_dtype,
                "requested_dtype": state.requested_dtype,
                "compatible_dtypes": state.compatible_dtypes,
                "summary": state.summary,
                "messages": [asdict(msg) for msg in state.messages],
                "artifacts": {name: asdict(artifact) for name, artifact in state.artifacts.items()},
                "total_messages_seen": state.total_messages_seen,
                "summary_refresh_count": state.summary_refresh_count,
                "cache_hit_count": state.cache_hit_count,
                "estimated_prompt_token_savings": state.estimated_prompt_token_savings,
                "conversion_count": state.conversion_count,
                "conversion_failures": state.conversion_failures,
                "last_conversion": state.last_conversion,
            }
        self.persistence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            raw = json.loads(self.persistence_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        for session_id, state in raw.items():
            session_state = SessionState(
                session_id=state.get("session_id", session_id),
                active_dtype=state.get("active_dtype", self.default_dtype),
                requested_dtype=state.get("requested_dtype", self.requested_dtype),
                compatible_dtypes=state.get("compatible_dtypes", self.compatible_dtypes),
                summary=state.get("summary", ""),
                total_messages_seen=state.get("total_messages_seen", 0),
                summary_refresh_count=state.get("summary_refresh_count", 0),
                cache_hit_count=state.get("cache_hit_count", 0),
                estimated_prompt_token_savings=state.get("estimated_prompt_token_savings", 0),
                conversion_count=state.get("conversion_count", 0),
                conversion_failures=state.get("conversion_failures", 0),
                last_conversion=state.get("last_conversion"),
            )
            for msg in state.get("messages", []):
                session_state.messages.append(
                    CachedMessage(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content", ""),
                        estimated_tokens=int(msg.get("estimated_tokens", 0)),
                        created_at_utc=msg.get("created_at_utc", self._now()),
                    )
                )
            for name, artifact in state.get("artifacts", {}).items():
                session_state.artifacts[name] = NumericContextArtifact(
                    name=name,
                    values=[float(v) for v in artifact.get("values", [])],
                    dtype=artifact.get("dtype", session_state.active_dtype),
                    updated_at_utc=artifact.get("updated_at_utc", self._now()),
                )
            self._sessions[session_id] = session_state
