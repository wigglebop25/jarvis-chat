"""
agent_logic/tool_handler.py
────────────────────────────
ToolHandlerMixin: tool execution, validation, and reprompt on failure.
"""
from __future__ import annotations
import logging
from typing import Any, Optional, TYPE_CHECKING

from ..models import MessageRole

if TYPE_CHECKING:
    from ..llm import ToolCall as ProviderToolCall, LLMProvider
    from ..message_builder import MessageBuilder
    from ..response_cache import LLMResponseCache
    from ..tool_discovery import ToolDiscovery
    from ..skills import ToolErrorRepromptSkill
    from ..mcp.router_protocol import MCPRouterLike

logger = logging.getLogger(__name__)


class ToolHandlerMixin:
    """Mixin for tool execution and definition management."""
    llm_provider: Optional[LLMProvider]
    tool_discovery: ToolDiscovery
    message_builder: MessageBuilder
    response_cache: Optional[LLMResponseCache]
    mcp_router: MCPRouterLike
    tool_error_reprompt_skill: ToolErrorRepromptSkill

    async def _refresh_tool_definitions(self: Any) -> None:
        supports_tools = bool(self.llm_provider and self.llm_provider.supports_tools)
        tool_defs_changed = await self.tool_discovery.refresh_async(supports_tools)
        if tool_defs_changed:
            self._tool_definitions = self.tool_discovery.tools
            self.message_builder.tool_definitions = self._tool_definitions
            if self.response_cache:
                self.response_cache.invalidate_all()

    def _refresh_tool_definitions_sync(self: Any) -> None:
        supports_tools = bool(self.llm_provider and self.llm_provider.supports_tools)
        tool_defs_changed = self.tool_discovery.refresh_sync(supports_tools)
        if tool_defs_changed:
            self._tool_definitions = self.tool_discovery.tools
            self.message_builder.tool_definitions = self._tool_definitions
            if self.response_cache:
                self.response_cache.invalidate_all()

    async def _execute_tool_with_reprompt(
        self: Any,
        tool_call: ProviderToolCall,
    ) -> tuple[ProviderToolCall, dict[str, Any]]:
        current_call = tool_call
        attempt = 0
        tools_payload = self._get_tools_payload()

        while True:
            # Phase 1: Validate tool parameters before execution
            if self._tool_definitions:
                from ..tools.validator import ToolParameterValidator
                tool_def = next(
                    (t for t in self._tool_definitions if t.get("name") == current_call.name),
                    None,
                )
                if tool_def:
                    is_valid, error_msg = ToolParameterValidator.validate_parameters(
                        tool_def, current_call.arguments
                    )
                    if not is_valid:
                        logger.warning(f"Tool validation failed for {current_call.name}: {error_msg}")
                        result_payload = {"is_error": True, "error": f"Validation Error: {error_msg}"}
                        attempt += 1
                        if attempt > self.tool_error_reprompt_skill.max_retries:
                            return current_call, result_payload

                        reprompt = self.tool_error_reprompt_skill.build_reprompt(
                            failed_tool_name=current_call.name,
                            failed_arguments=current_call.arguments,
                            error_message=error_msg or "invalid parameters",
                            attempt=attempt,
                        )
                        self._record_message(MessageRole.SYSTEM, reprompt)
                        if not self.llm_provider:
                            raise RuntimeError("LLM provider not initialized for tool repair")
                        repair_response = await self.llm_provider.complete(
                            self._build_messages(), tools=tools_payload
                        )
                        if repair_response.tool_calls:
                            current_call = repair_response.tool_calls[0]
                            continue
                        return current_call, result_payload

            args = {k: v for k, v in current_call.arguments.items() if k != "name"}
            result = await self.mcp_router.execute_tool(current_call.name, **args)
            result_payload = result.model_dump()
            if not result.is_error:
                return current_call, result_payload

            attempt += 1
            if attempt > self.tool_error_reprompt_skill.max_retries:
                return current_call, result_payload

            reprompt = self.tool_error_reprompt_skill.build_reprompt(
                failed_tool_name=current_call.name,
                failed_arguments=current_call.arguments,
                error_message=result.error or "unknown tool error",
                attempt=attempt,
            )
            self._record_message(MessageRole.SYSTEM, reprompt)
            self.tool_error_reprompt_skill.backoff(attempt)

            if not self.llm_provider:
                raise RuntimeError("LLM provider not initialized for tool repair")
            repair_response = await self.llm_provider.complete(
                self._build_messages(), tools=tools_payload
            )
            if repair_response.tool_calls:
                current_call = repair_response.tool_calls[0]
                continue

            if repair_response.text:
                self._record_message(MessageRole.ASSISTANT, repair_response.text)
            return current_call, result_payload
