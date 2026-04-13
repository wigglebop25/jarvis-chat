"""
Skill helper for recovering from failed tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time


@dataclass(slots=True)
class ToolErrorRepromptSkill:
    """
    Build deterministic reprompts for failed tool-call retries.
    """

    max_retries: int = 2
    base_backoff_seconds: float = 0.5

    def build_reprompt(
        self,
        *,
        failed_tool_name: str,
        failed_arguments: dict,
        error_message: str,
        attempt: int,
    ) -> str:
        arguments_json = json.dumps(failed_arguments, sort_keys=True)
        return (
            "Tool execution failed.\n"
            f"Attempt: {attempt}/{self.max_retries}\n"
            f"Failed tool: {failed_tool_name}\n"
            f"Failed arguments: {arguments_json}\n"
            f"Error: {error_message}\n\n"
            "Generate exactly one corrected tool call with valid arguments. "
            "Do not repeat invalid arguments. "
            "If correction is impossible, return a short assistant message explaining the failure."
        )

    def backoff(self, attempt: int) -> None:
        if self.base_backoff_seconds <= 0:
            return
        # Linear backoff keeps retries deterministic and bounded.
        time.sleep(self.base_backoff_seconds * attempt)
