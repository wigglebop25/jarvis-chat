from typing import Any, Optional
import os
import asyncio
import logging

import httpx

from .circuit_breaker import CircuitBreaker, CircuitBreakerError
from ..tools.retry_utils import async_retry_with_backoff, MCPRateLimitError, MCPTimeoutError

logger = logging.getLogger(__name__)


def _serialize_protobuf(obj: Any) -> Any:
    """Convert protobuf objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): _serialize_protobuf(v) for k, v in obj.items()}

    if hasattr(obj, "ListFields"):
        # Protobuf message: ListFields returns (FieldDescriptor, value) tuples.
        return {
            getattr(field, "name", str(field)): _serialize_protobuf(value)
            for field, value in obj.ListFields()
        }

    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        # Iterable (list, repeated field, etc.)
        try:
            return [_serialize_protobuf(item) for item in obj]
        except TypeError:
            return obj

    return obj


class MCPClient:
    """HTTP client for JSON-RPC communication with MCP server."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        resolved_base_url: str = base_url or os.getenv("MCP_SERVER_URL") or "http://127.0.0.1:5050"
        self.base_url = resolved_base_url.rstrip("/")
        
        # Decide on the final endpoint. 
        # If the URL already ends with /mcp, /jsonrpc, or similar, use it as is.
        if any(self.base_url.endswith(suffix) for suffix in ["/mcp", "/jsonrpc", "/rpc"]):
            self.endpoint = self.base_url
        else:
            # Default to appending /jsonrpc for backward compatibility
            self.endpoint = f"{self.base_url}/jsonrpc"
            
        self.timeout_seconds = timeout
        request_timeout = httpx.Timeout(timeout, connect=min(5.0, float(timeout)))
        self.client = httpx.AsyncClient(
            timeout=request_timeout,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        self._request_id = 0
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Initialize circuit breaker per endpoint
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60,
            name=f"MCPServer({resolved_base_url})",
        )

    async def initialize(self) -> None:
        """Perform the MCP initialization handshake."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            logger.info(f"Initializing MCP HTTP client: {self.endpoint}")
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "jarvis-chat",
                    "version": "0.1.0"
                }
            }
            
            # Send initialize request (directly using _make_rpc_request to avoid loop)
            await self._make_rpc_request("initialize", params)

            # Send initialized notification if required by the protocol

            # JSON-RPC notifications have no ID
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
            try:
                await self.client.post(self.endpoint, json=notification)
            except Exception as e:
                logger.debug(f"Failed to send initialized notification: {e}")

            self._initialized = True
            logger.info(f"MCP HTTP client initialized: {self.base_url}")

    async def call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Call a method on the MCP server with timeout and rate limit protection.
        """
        # Ensure client is initialized before first call
        if not self._initialized and method != "initialize":
            await self.initialize()

        try:
            return await self.circuit_breaker.call(
                self._execute_call,
                method,
                params,
                timeout,
            )
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker prevented call to '{method}': {e}")
            raise
    
    async def _execute_call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Execute the actual RPC call (called through circuit breaker)."""
        return await async_retry_with_backoff(
            self._make_rpc_request,
            method,
            params,
            timeout,
            max_retries=4,
            base_delay=1.0,
            backoff_factor=2.0,
        )
    
    async def _make_rpc_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Make a single RPC request to the MCP server."""
        timeout_seconds = timeout or self.timeout_seconds
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": str(self._request_id),
            "method": method,
            "params": params or {},
        }
        
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await self.client.post(
                    self.endpoint,
                    json=request,
                    timeout=timeout_seconds,
                )
                
                # Handle rate limit (429)
                if response.status_code == 429:
                    raise MCPRateLimitError(
                        f"Rate limit exceeded for '{method}'"
                    )
                
                response.raise_for_status()

                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Invalid MCP response shape")
                error = data.get("error")
                if error:
                    # Check for "initialize request required" specifically to provide better diagnostics
                    if error.get("code") == -32600 and "initialize" in error.get("message", "").lower():
                        logger.error("Server rejected request: initialization required.")
                    
                    raise RuntimeError(
                        f"RPC error ({error.get('code')}): "
                        f"{error.get('message')}"
                    )

                result = data.get("result")
                if result is None:
                    result = {}
                return _serialize_protobuf(result)

        except asyncio.TimeoutError as e:
            error_msg = f"MCP call to '{method}' timed out after {timeout_seconds}s"
            raise MCPTimeoutError(error_msg) from e
        except httpx.HTTPError as e:
            raise ConnectionError(f"MCP request failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout_seconds,
                )
                return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
