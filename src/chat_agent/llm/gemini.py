import os
import warnings
from typing import Any, AsyncGenerator, Optional

# Suppress FutureWarning from deprecated google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)

from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerationConfig

from .base import LLMProvider, LLMConfigurationError, LLMProviderError, LLMResponse, ToolCall

import logging
logger = logging.getLogger(__name__)


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
        configure(api_key=api_key)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature or 0.7
        self.max_tokens = max_tokens or 2048
        self.client = GenerativeModel(model)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        return True

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_available_models(self) -> list[str]:
        try:
            # Use the properly exported list_models from the models module
            from google.generativeai.models import list_models
            models = list_models()
            available = []
            for m in models:
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    name = m.name.replace("models/", "")
                    available.append(name)
            return available if available else [self.model]
        except Exception:
            return [self.model]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            # GenerativeModel.count_tokens can take a string
            return self.client.count_tokens(text).total_tokens
        except Exception:
            return len(text) // 4

    def _to_contents(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            # Map system to user for Gemini if not using system_instruction
            mapped_role = "user" if role in {"user", "system"} else "model"
            
            content_text = msg.get("content", "")
            if not content_text and role != "tool":
                continue

            # Merge consecutive messages with the same role
            if contents and contents[-1]["role"] == mapped_role:
                contents[-1]["parts"][0]["text"] += "\n\n" + content_text
            else:
                contents.append(
                    {
                        "role": mapped_role,
                        "parts": [{"text": content_text}],
                    }
                )
        return contents

    def _convert_tools_to_gemini(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert tool definitions to Gemini format (single tools object)."""
        from ..tools.schemas import ToolSchemaConverter
        
        function_declarations = []
        for tool in tools:
            function_declarations.append(ToolSchemaConverter.to_gemini(tool))
        
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
                "generation_config": GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            }
            
            gemini_tools = None
            if tools and self.supports_tools:
                gemini_tools = self._convert_tools_to_gemini(tools)
                kwargs["tools"] = gemini_tools
            
            contents = self._to_contents(messages)
            client = self.client
            cache = None
            
            # Context Caching for large prompts
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > 80000 and len(contents) > 1:  # Rough heuristic before counting
                total_tokens = sum(self.count_tokens(m.get("content", "")) for m in messages)
                if total_tokens >= 32768:
                    try:
                        from google.generativeai import caching
                        import datetime
                        cache = caching.CachedContent.create(
                            model=self.model,
                            contents=contents[:-1],
                            tools=gemini_tools,
                            ttl=datetime.timedelta(minutes=5),
                        )
                        client = GenerativeModel.from_cached_content(cached_content=cache)
                        contents = [contents[-1]]
                    except Exception as e:
                        pass # Fallback to normal execution

            try:
                response = client.generate_content(contents, **kwargs)
            finally:
                if cache:
                    try:
                        cache.delete()
                    except Exception:
                        pass

            tool_calls = self._extract_tool_calls(response) if tools else []
            
            return LLMResponse(
                text=self._extract_text(response),
                tool_calls=tool_calls,
                model=self.model,
                usage=self._usage_from_response(response),
            )
        except Exception as e:
            import traceback
            logger.error(f"Gemini API Error: {e}\n{traceback.format_exc()}")
            raise LLMProviderError(f"Gemini request failed: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            kwargs = {
                "stream": False,
                "generation_config": GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            }
            
            gemini_tools = None
            if tools and self.supports_tools:
                gemini_tools = self._convert_tools_to_gemini(tools)
                kwargs["tools"] = gemini_tools
            
            contents = self._to_contents(messages)
            client = self.client
            cache = None
            
            # Context Caching for large prompts
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > 80000 and len(contents) > 1:
                total_tokens = sum(self.count_tokens(m.get("content", "")) for m in messages)
                if total_tokens >= 32768:
                    try:
                        from google.generativeai import caching
                        import datetime
                        cache = caching.CachedContent.create(
                            model=self.model,
                            contents=contents[:-1],
                            tools=gemini_tools,
                            ttl=datetime.timedelta(minutes=5),
                        )
                        client = GenerativeModel.from_cached_content(cached_content=cache)
                        contents = [contents[-1]]
                    except Exception as e:
                        pass

            try:
                response = await client.generate_content_async(contents, **kwargs)
            finally:
                if cache:
                    try:
                        cache.delete()
                    except Exception:
                        pass
            
            tool_calls = self._extract_tool_calls(response) if tools else []
            
            return LLMResponse(
                text=self._extract_text(response),
                tool_calls=tool_calls,
                model=self.model,
                usage=self._usage_from_response(response),
            )
        except Exception as e:
            import traceback
            logger.error(f"Gemini API Error: {e}\n{traceback.format_exc()}")
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
                generation_config=GenerationConfig(
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
