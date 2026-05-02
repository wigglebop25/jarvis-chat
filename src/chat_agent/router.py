"""
MCP Router

Routes tool calls to the MCP Server via HTTP/JSON-RPC.
This module bridges the Chat Agent and MCP Server.

Communication uses HTTP with JSON-RPC 2.0 protocol to avoid
STDOUT noise issues that come with stdio-based transports.
"""

import asyncio
import logging
import uuid
from typing import Optional

import httpx
from .config import MCPConfig
from .models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class MCPRouter:
    """
    Routes tool calls to the MCP Server via HTTP/JSON-RPC.
    """
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """
        Initialize the MCP Router.
        
        Args:
            config: MCP Server configuration
        """
        self.config = config or MCPConfig()
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call via JSON-RPC and return the result.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            ToolResult with success status and result/error
        """
        logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")

        return await self._execute_jsonrpc(tool_call)
    
    def execute_tool_sync(self, tool_call: ToolCall) -> ToolResult:
        """Synchronous version of execute_tool."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.execute_tool(tool_call))
    
    async def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        tasks = [self.execute_tool(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)
    
    async def list_tools(self) -> list[dict]:
        """Fetch available tools from MCP Server."""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/list",
                "params": {},
            }

            response = await self._client.post(
                f"{self.config.url}/jsonrpc",
                json=request,
            )

            if response.status_code == 200:
                data = response.json()
                if "result" in data and "tools" in data["result"]:
                    return data["result"]["tools"]
                return []
                    
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
        
        return []
    
    async def _execute_jsonrpc(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call via HTTP JSON-RPC to the MCP Server."""
        request = {
            "jsonrpc": "2.0",
            "id": tool_call.id,
            "method": "tools/call",
            "params": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        }
        
        url = f"{self.config.url}/jsonrpc"
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self._client.post(
                    url,
                    json=request,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data and data["error"]:
                        return ToolResult(
                            tool_call_id=tool_call.id,
                            success=False,
                            error=data["error"].get("message", "Unknown error")
                        )
                    
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        success=True,
                        result=data.get("result")
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"MCP Server error: {error_msg}")
                    
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        error=error_msg
                    )
                        
            except httpx.TimeoutException:
                logger.warning(f"Timeout calling MCP Server (attempt {attempt + 1})")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error="MCP Server timeout"
                )
                
            except httpx.ConnectError:
                logger.warning(f"Cannot connect to MCP Server at {url}")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=f"Cannot connect to MCP Server at {self.config.url}"
                )
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=str(e)
                )
        
        return ToolResult(
            tool_call_id=tool_call.id,
            success=False,
            error="Max retry attempts exceeded"
        )
    
    async def health_check(self) -> bool:
        """Check if the MCP Server is healthy."""
        try:
            response = await self._client.get(f"{self.config.url}/health", timeout=5.0)
            return response.status_code == 200
                
        except Exception:
            return False


def create_router(config: Optional[MCPConfig] = None) -> MCPRouter:
    """Factory function to create an MCP Router."""
    return MCPRouter(config)
