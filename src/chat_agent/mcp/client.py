from typing import Any, Optional
import os

import httpx


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
        resolved_base_url = base_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:5050")
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = timeout
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

    async def call(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
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
        request = {
            "jsonrpc": "2.0",
            "id": str(self._request_id),
            "method": method,
            "params": params or {},
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/jsonrpc",
                json=request,
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("Invalid MCP response shape")
            error = data.get("error")
            if error:
                raise RuntimeError(
                    f"RPC error ({error.get('code')}): "
                    f"{error.get('message')}"
                )

            # Serialize protobuf objects to JSON-safe format
            result = data.get("result")
            if result is None:
                result = {}
            return _serialize_protobuf(result)

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
