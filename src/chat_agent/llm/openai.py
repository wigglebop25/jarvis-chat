import json
import os
from typing import Any, AsyncGenerator, Optional

from openai import AsyncOpenAI, OpenAI

from .base import LLMProvider, LLMConfigurationError, LLMProviderError, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY environment variable not set")

        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supports_tools(self) -> bool:
        return True

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate synchronous completion using OpenAI."""
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            if tools:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                    for tool in tools
                ]

            response = self.client.chat.completions.create(**kwargs)

            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls or []:
                if tool_call.type == "function":
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=args,
                        )
                    )

            return LLMResponse(
                text=response.choices[0].message.content or "",
                tool_calls=tool_calls,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI request failed: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate asynchronous completion using OpenAI."""
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            if tools:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                    for tool in tools
                ]

            response = await self.async_client.chat.completions.create(**kwargs)

            tool_calls = []
            for tool_call in response.choices[0].message.tool_calls or []:
                if tool_call.type == "function":
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=args,
                        )
                    )

            return LLMResponse(
                text=response.choices[0].message.content or "",
                tool_calls=tool_calls,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        except Exception as e:
            raise LLMProviderError(f"OpenAI request failed: {e}") from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from OpenAI."""
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            if tools:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                    for tool in tools
                ]

            async with self.async_client.chat.completions.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as e:
            raise LLMProviderError(f"OpenAI streaming failed: {e}") from e
