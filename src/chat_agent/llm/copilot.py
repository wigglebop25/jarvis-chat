import os
import shutil
import subprocess
from typing import Any, AsyncGenerator, Optional

from .base import LLMProvider, LLMResponse, LLMProviderError, LLMConfigurationError


class CopilotProvider(LLMProvider):
    """GitHub Copilot provider using Copilot CLI."""

    def __init__(self, model: str = "claude-haiku-4.5", temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Initialize Copilot provider.
        
        Args:
            model: Model name (claude-haiku-4.5, claude-sonnet-4, etc.)
            temperature: Response temperature (0.0-2.0)
            max_tokens: Max tokens in response
        """
        self.model = model
        self.temperature = temperature or 0.7
        self.max_tokens = max_tokens or 2048
        
        # Find copilot CLI executable
        copilot_path = shutil.which("copilot")
        if not copilot_path:
            raise LLMConfigurationError(
                "Copilot CLI not found. Install from: https://github.com/github/copilot-cli"
            )
        self.copilot_path: str = copilot_path
        
        # Verify Copilot CLI is working
        try:
            result = subprocess.run(
                [self.copilot_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise LLMConfigurationError(
                    "Copilot CLI not working. Try: copilot auth login"
                )
        except subprocess.TimeoutExpired:
            raise LLMConfigurationError(
                "Copilot CLI check timed out."
            )
        except Exception as e:
            raise LLMConfigurationError(f"Copilot CLI check failed: {e}")

    @property
    def name(self) -> str:
        return "copilot"

    @property
    def supports_tools(self) -> bool:
        return True

    def is_configured(self) -> bool:
        try:
            result = subprocess.run(
                [self.copilot_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate synchronous completion using Copilot CLI."""
        try:
            # Format messages for copilot CLI
            message_text = "\n".join([msg.get("content", "") for msg in messages])
            
            # Call copilot CLI with prompt mode
            result = subprocess.run(
                [self.copilot_path, "-p", message_text, "--allow-all-tools", "-s"],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ}
            )
            
            if result.returncode != 0:
                # Fall back to stderr if available
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise LLMProviderError(f"Copilot CLI error: {error_msg}")
            
            response_text = result.stdout.strip()
            if not response_text:
                raise LLMProviderError("Copilot CLI returned empty response")
            
            return LLMResponse(
                text=response_text,
                tool_calls=[],
                model=self.model,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )
        except subprocess.TimeoutExpired:
            raise LLMProviderError("Copilot request timed out (>30s)")
        except Exception as e:
            raise LLMProviderError(f"Copilot request failed: {e}") from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate asynchronous completion using GitHub Copilot CLI."""
        # For now, use sync version (Copilot CLI is inherently sync)
        return self.complete_sync(messages, tools)

    async def stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion from Copilot CLI."""
        try:
            # Format messages for copilot CLI
            message_text = "\n".join([msg.get("content", "") for msg in messages])
            
            # Call copilot CLI with streaming (without -s flag to get output)
            process = subprocess.Popen(
                [self.copilot_path, "-p", message_text, "--allow-all-tools"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ}
            )
            
            # Stream output line by line
            if process.stdout:
                for line in process.stdout:
                    if line.strip():
                        yield line
            
            process.wait(timeout=30)
            if process.returncode != 0:
                raise LLMProviderError("Copilot streaming failed")
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise LLMProviderError("Copilot streaming timed out")
        except Exception as e:
            raise LLMProviderError(f"Copilot streaming failed: {e}") from e
