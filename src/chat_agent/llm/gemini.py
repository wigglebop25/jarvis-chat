import os
import warnings
from typing import Any, AsyncGenerator, Optional

# Suppress FutureWarning from deprecated google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai

from .base import LLMProvider, LLMConfigurationError, LLMProviderError, LLMResponse, ToolCall


class GeminiProvider(LLMProvider):
    """Google Gemini / AI Studio API provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        # Set API key via environment or direct parameter
        genai.api_key = api_key
        self.api_key = api_key
        self.model = model
        self.temperature = temperature or 0.7
        self.max_tokens = max_tokens or 2048
        self.client = genai.GenerativeModel(model)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        return True

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _to_contents(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            mapped_role = "user" if role in {"user", "system"} else "model"
            contents.append(
                {
                    "role": mapped_role,
                    "parts": [{"text": msg.get("content", "")}],
                }
            )
        return contents

    def _convert_tools_to_gemini(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert tool definitions to Gemini format (single tools object)."""
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
        
        return {
            "function_declarations": function_declarations
        }

    def _to_plain_value(self, value: Any) -> Any:
        """Convert protobuf/map/repeated values into plain Python JSON-safe values."""
        if isinstance(value, dict):
            return {str(k): self._to_plain_value(v) for k, v in value.items()}

        if hasattr(value, "ListFields"):
            return {
                getattr(field, "name", str(field)): self._to_plain_value(field_value)
                for field, field_value in value.ListFields()
            }

        if hasattr(value, "items") and not isinstance(value, (str, bytes)):
            try:
                return {str(k): self._to_plain_value(v) for k, v in value.items()}
            except TypeError:
                pass

        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            try:
                return [self._to_plain_value(item) for item in value]
            except TypeError:
                pass

        return value

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract tool calls from Gemini response."""
        tool_calls = []
        
        # Check if response has candidates with function calls
        if not hasattr(response, "candidates") or not response.candidates:
            return tool_calls
        
        call_id = 0
        for candidate in response.candidates:
            if not hasattr(candidate, "content") or not candidate.content:
                continue
            
            if not hasattr(candidate.content, "parts"):
                continue
            
            for part in candidate.content.parts:
                # Check if part is a function call
                if hasattr(part, "function_call") and part.function_call:
                    func_call = part.function_call
                    call_id += 1
                    
                    # Extract arguments - handle both protobuf and dict forms
                    arguments = {}
                    if hasattr(func_call, "args"):
                        plain_args = self._to_plain_value(func_call.args)
                        if isinstance(plain_args, dict):
                            arguments = plain_args
                    
                    tool_calls.append(
                        ToolCall(
                            id=str(call_id),
                            name=getattr(func_call, "name", ""),
                            arguments=arguments,
                        )
                    )
        
        return tool_calls

    def _extract_text(self, response: Any) -> str:
        """Extract text parts safely without touching response.text."""
        texts: list[str] = []

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)

        return "\n".join(texts).strip()

    def _usage_from_response(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        usage_dict = usage if isinstance(usage, dict) else {}

        prompt_tokens = int(
            getattr(usage, "prompt_token_count", 0)
            or usage_dict.get("prompt_token_count", 0)
            or 0
        )
        completion_tokens = int(
            getattr(usage, "candidates_token_count", 0)
            or usage_dict.get("candidates_token_count", 0)
            or usage_dict.get("output_token_count", 0)
            or 0
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            kwargs = {
                "stream": False,
                "generation_config": genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            }
            
            # Add tools if provided
            if tools and self.supports_tools:
                gemini_tools = self._convert_tools_to_gemini(tools)
                kwargs["tools"] = gemini_tools
            
            response = self.client.generate_content(
                self._to_contents(messages),
                **kwargs,
            )
            
            # Extract tool calls if any
            tool_calls = self._extract_tool_calls(response) if tools else []
            
            return LLMResponse(
                text=self._extract_text(response),
                tool_calls=tool_calls,
                model=self.model,
                usage=self._usage_from_response(response),
            )
        except Exception as e:
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            kwargs = {
                "stream": False,
                "generation_config": genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            }
            
            # Add tools if provided
            if tools and self.supports_tools:
                gemini_tools = self._convert_tools_to_gemini(tools)
                kwargs["tools"] = gemini_tools
            
            response = await self.client.generate_content_async(
                self._to_contents(messages),
                **kwargs,
            )
            
            # Extract tool calls if any
            tool_calls = self._extract_tool_calls(response) if tools else []
            
            return LLMResponse(
                text=self._extract_text(response),
                tool_calls=tool_calls,
                model=self.model,
                usage=self._usage_from_response(response),
            )
        except Exception as e:
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            stream_response = await self.client.generate_content_async(
                self._to_contents(messages),
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            async for chunk in stream_response:
                text = self._extract_text(chunk)
                if text:
                    yield text
        except Exception as e:
            raise LLMProviderError(f"Gemini streaming failed: {e}") from e
