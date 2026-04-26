import asyncio
import time
import json
import httpx
from src.chat_agent.intent import recognize_intent, get_tool_name_for_intent, map_intent_params_to_tool
from src.chat_agent.mcp.router import MCPRouter

queries = [
    "What is my CPU usage?",
    "Volume up",
    "List folders in D drive",
    "Turn off the wifi",
    "Organize my downloads folder",
    "Play some music"
]

async def benchmark_old_python():
    print("--- Python Intent Routing ---")
    latencies = []
    
    for q in queries:
        t0 = time.perf_counter()
        
        # Original flow
        intent = recognize_intent(q)
        tool_name = get_tool_name_for_intent(intent)
        params = map_intent_params_to_tool(intent) if intent.type.value != "unknown" else {}
        
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        latencies.append(lat_ms)
        print(f"[{lat_ms:.2f} ms] '{q}' -> {intent.type.value} -> {tool_name} with {params}")
    
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    print(f"\nPython P50: {p50:.2f} ms | P95: {p95:.2f} ms")


async def benchmark_new_rust():
    print("\n--- Rust MCP Intent Routing ---")
    from src.chat_agent.mcp.client import MCPClient
    client = MCPClient("http://127.0.0.1:5050")
    router = MCPRouter(mcp_client=client)
    latencies = []
    
    # Warmup request
    try:
        await router.route_and_call("warmup")
    except Exception as e:
        print("Warmup failed, Make sure the Rust MCP server is running at http://127.0.0.1:5050!")
        print(f"Error: {e}")
        return

    for q in queries:
        t0 = time.perf_counter()
        
        # New flow (route only for fair comparison of intent matching + params parsing)
        # Using the jsonrpc directly to measure just the routing overhead, not execution.
        try:
            resp = await router.mcp_client.call("jarvis/route", {"text": q})
        except Exception as e:
            print(f"Failed to route '{q}': {e}")
            continue
            
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        latencies.append(lat_ms)
        
        intent = resp.get("intent", "unknown")
        tool_name = resp.get("tool_name")
        params = resp.get("arguments", {})
        
        print(f"[{lat_ms:.2f} ms] '{q}' -> {intent} -> {tool_name} with {params}")
    
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        print(f"\nRust P50: {p50:.2f} ms | P95: {p95:.2f} ms")

async def main():
    print("Running Phase 1 Latency Benchmark...\n")
    await benchmark_old_python()
    await benchmark_new_rust()

if __name__ == "__main__":
    asyncio.run(main())
