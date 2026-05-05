from .agent import ChatAgent
from .langgraph_agent import LangGraphChatAgent
from .config import AgentConfig, LLMConfig, OpenAIConfig
from .intent import get_tool_name_for_intent, map_intent_params_to_tool, recognize_intent
from .llm import LLMProvider, LLMResponse, ToolCall
from .models import AgentResponse, IntentType, MessageRole, ToolCall as AgentToolCall


class _CompatChatAgent(ChatAgent):
    """
    Backward-compatible synchronous API used by legacy tests/callers.

    The modern ChatAgent API is async and tool/LLM-driven. Legacy callers
    expect a synchronous `process_transcript()` returning AgentResponse.
    """

    def process_transcript(self, transcript: str) -> AgentResponse:  # type: ignore[override]
        cleaned = (transcript or "").strip()
        intent = recognize_intent(transcript)

        if not cleaned:
            response_text = "I didn't catch that. Could you please repeat?"
        elif intent.type == IntentType.UNKNOWN:
            response_text = "I don't understand. Could you rephrase that?"
        else:
            response_text = f"Recognized: {intent.type.value} (confidence: {intent.confidence:.2f})"

        tool_calls: list[AgentToolCall] = []
        tool_name = get_tool_name_for_intent(intent)
        if tool_name:
            tool_calls.append(
                AgentToolCall(
                    id=f"intent-{len(self.context.messages)}",
                    name=tool_name,
                    arguments=map_intent_params_to_tool(intent),
                )
            )

        self._record_message(MessageRole.USER, transcript)
        self._record_message(MessageRole.ASSISTANT, response_text)

        return AgentResponse(
            text=response_text,
            intent=intent,
            tool_calls=tool_calls,
        )


def create_agent(config: AgentConfig | None = None, llm_config: LLMConfig | None = None) -> ChatAgent:
    """Create a backward-compatible ChatAgent instance."""
    return _CompatChatAgent(config=config, llm_config=llm_config)


__all__ = [
    "ChatAgent",
    "LangGraphChatAgent",
    "create_agent",
    "recognize_intent",
    "IntentType",
    "AgentConfig",
    "LLMConfig",
    "OpenAIConfig",
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
]
