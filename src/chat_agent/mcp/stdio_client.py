from typing import Any, Optional, cast
import os
import json
import logging
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CustomRequest(BaseModel):
    """Generic request model for custom MCP methods to avoid jsonrpc collisions."""
    method: str
    params: dict[str, Any]

class CustomResult(BaseModel):
    """Wrapper to safely receive dictionary results from custom MCP methods."""
    model_config = {"extra": "allow"}

class StdioMCPClient:
    """Stdio-based client for MCP communication using the official SDK."""

    def __init__(self, command: str, args: list[str], timeout: int = 30):
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=os.environ.copy()
        )
        self.timeout = timeout
        self.session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()
        self._request_id = 0
        self._context_manager = None

    async def connect(self):
        """Initialize the stdio connection and session."""
        async with self._lock:
            if self.session:
                return

            logger.info(f"Connecting to MCP server: {self.server_params.command} {' '.join(self.server_params.args)}")
            
            self._context_manager = stdio_client(self.server_params)
            self._read, self._write = await self._context_manager.__aenter__()
            self.session = ClientSession(self._read, self._write)
            await self.session.__aenter__()
            await self.session.initialize()
            logger.info("MCP Stdio session initialized.")

    async def call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Call a method on the MCP server.
        Note: The official SDK maps tools/call to session.call_tool
        """
        if not self.session:
            await self.connect()
        
        # Ensure session is active after connect
        assert self.session is not None

        if method == "tools/call":
            safe_params = params or {}
            tool_name = safe_params.get("name")
            if not isinstance(tool_name, str):
                logger.error(f"[stdio_client] Missing tool name in params: {safe_params}")
                raise ValueError("Tool name must be a string")
                
            arguments = safe_params.get("arguments", {})
            logger.debug(f"[stdio_client] Tool Call: {tool_name} with {json.dumps(arguments)}")
            
            result = await self.session.call_tool(tool_name, arguments)
            
            content = []
            for item in result.content:
                # Use getattr to safely access 'text' and satisfy Pylance
                text = getattr(item, "text", None)
                if text is not None:
                    content.append({"type": "text", "text": text})
            
            if len(content) == 1 and content[0]["type"] == "text":
                try:
                    return json.loads(content[0]["text"])
                except Exception:
                    return content[0]["text"]
            
            return content

        elif method == "tools/list":
            logger.debug("[stdio_client] Fetching tool list...")
            result = await self.session.list_tools()
            
            tool_defs = []
            for t in result.tools:
                if hasattr(t, "model_dump"):
                    tool_defs.append(t.model_dump())
                elif isinstance(t, dict):
                    tool_defs.append(t)
                else:
                    try:
                        tool_defs.append(dict(t))
                    except Exception:
                        pass
            
            return {"tools": tool_defs}

        # Generic call for other methods (like jarvis/route_and_call)
        self._request_id += 1
        
        # We use a CustomRequest model that only has 'method' and 'params'.
        # The SDK's send_request then wraps this in a JSONRPCRequest(jsonrpc="2.0", id=..., **model_dump())
        # This prevents the 'got multiple values for keyword argument jsonrpc' error.
        request = CustomRequest(method=method, params=params or {})
        
        logger.debug(f"[stdio_client] Custom Request: {method}")
        
        # Use CustomResult to handle arbitrary dictionary responses
        # The SDK's send_request handles the casting internally if we provide a BaseModel
        response = await self.session.send_request(cast(Any, request), result_type=CustomResult)
        return response.model_dump()

    async def health_check(self) -> bool:
        """Check if MCP server is responsive."""
        try:
            if not self.session:
                return False
            await self.session.list_tools()
            return True
        except Exception:
            return False

    async def close(self):
        """Close the stdio connection."""
        async with self._lock:
            if self.session:
                await self.session.__aexit__(None, None, None)
                if self._context_manager:
                    await self._context_manager.__aexit__(None, None, None)
                self.session = None
                self._context_manager = None
                logger.info("MCP Stdio session closed.")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
