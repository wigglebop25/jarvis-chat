import os
from typing import Any, AsyncGenerator, Optional

import google.generativeai as genai
from google.ai.generativelanguage import Content, Part

from .base import LLMProvider, LLMConfigurationError, LLMProviderError, LLMResponse, ToolCall


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model = model
        self.client = genai.GenerativeModel(
            model,
            safety_settings=[
                {"category": "HARM_CATEGORY_UNSPECIFIED", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )

    @property
    def name(self) -> str:
        return "gemini"

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
        """Generate synchronous completion using Gemini."""
        try:
            contents = []
            for msg in messages:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(Content(role=role, parts=[Part.from_text(msg.get("content", ""))]))

            tool_config = None
            if tools:
                tool_config = {
                    "function_declarations": [
                        {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                        for tool in tools
                    ]
                }

            response = self.client.generate_content(
                contents,
                tools=[tool_config] if tool_config else None,
                stream=False,
            )

            tool_calls = []
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        tool_calls.append(
                            ToolCall(
                                id="",
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args),
                            )
                        )

            return LLMResponse(
                text=response.text or "",
                tool_calls=tool_calls,
                model=self.model,
                usage={
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                },
            )
        except Exception as e:
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate asynchronous completion using Gemini."""
        try:
            contents = []
            for msg in messages:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(Content(role=role, parts=[Part.from_text(msg.get("content", ""))]))

            tool_config = None
            if tools:
                tool_config = {
                    "function_declarations": [
                        {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                        for tool in tools
                    ]
                }

            response = await self.client.generate_content_async(
                contents,
                tools=[tool_config] if tool_config else None,
                stream=False,
            )

            tool_calls = []
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        tool_calls.append(
                            ToolCall(
                                id="",
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args),
                            )
                        )

            return LLMResponse(
                text=response.text or "",
                tool_calls=tool_calls,
                model=self.model,
                usage={
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                },
            )
        except Exception as e:
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Gemini."""
        try:
            contents = []
            for msg in messages:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(Content(role=role, parts=[Part.from_text(msg.get("content", ""))]))

            tool_config = None
            if tools:
                tool_config = {
                    "function_declarations": [
                        {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                        for tool in tools
                    ]
                }

            async for chunk in await self.client.generate_content_async(
                contents,
                tools=[tool_config] if tool_config else None,
                stream=True,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMProviderError(f"Gemini streaming failed: {e}") from e
