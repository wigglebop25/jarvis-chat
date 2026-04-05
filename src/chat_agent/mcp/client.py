from typing import Any, Optional

import httpx

from .models import MCPRequest, MCPResponse


class MCPClient:
    """HTTP client for JSON-RPC communication with MCP server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._request_id = 0

    async def call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Call a method on the MCP server.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout (uses client default if not specified)

        Returns:
            Result from the RPC call
        """
        self._request_id += 1
        request = MCPRequest(
            id=self._request_id,
            method=method,
            params=params or {},
        )

        try:
            response = await self.client.post(
                f"{self.base_url}/rpc",
                json=request.model_dump(),
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            rpc_response = MCPResponse(**data)

            if rpc_response.error:
                raise RuntimeError(
                    f"RPC error ({rpc_response.error.get('code')}): "
                    f"{rpc_response.error.get('message')}"
                )

            return rpc_response.result or {}

        except httpx.HTTPError as e:
            raise RuntimeError(f"MCP request failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
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
