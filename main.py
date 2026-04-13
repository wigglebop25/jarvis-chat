"""
JARVIS Chat Agent CLI

Simple command-line interface for testing the Chat Agent.

Usage:
    uv run main.py                  # Interactive mode
    uv run main.py --server         # Server mode (HTTP API)
    uv run main.py "command"        # Single command
    uv run main.py --server "cmd"   # Single command (server mode)
"""

import sys
import os
import argparse
import json
import asyncio
import threading
from typing import Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chat_agent import ChatAgent, AgentConfig
from chat_agent.mcp import MCPRouter

# Global event loop for persistent async operations
_event_loop = None
_loop_lock = threading.Lock()

def get_event_loop():
    """Get or create a persistent event loop."""
    global _event_loop
    with _loop_lock:
        if _event_loop is None or _event_loop.is_closed():
            try:
                _event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_event_loop)
            except RuntimeError:
                _event_loop = asyncio.get_event_loop()
        return _event_loop

def run_async(coro):
    """Run an async coroutine using the persistent event loop."""
    loop = get_event_loop()
    if loop.is_running():
        # If loop is already running, we can't use run_until_complete
        # This shouldn't happen in our case, but handle it gracefully
        raise RuntimeError("Event loop is already running")
    return loop.run_until_complete(coro)


def main():
    """Main entry point for the Chat Agent CLI."""
    parser = argparse.ArgumentParser(description="JARVIS Chat Agent CLI")
    parser.add_argument("--server", "-s", action="store_true", 
                        help="Start HTTP API server (default: interactive mode)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="HTTP server port (default: 8000)")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:5050",
                        help="MCP server URL (default: http://127.0.0.1:5050)")
    parser.add_argument("command", nargs="*", help="Command to execute")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("JARVIS Chat Agent v0.1.0")
    print("=" * 50)
    
    agent = ChatAgent(AgentConfig())
    
    if args.server:
        # Start HTTP server
        start_http_server(agent, args.port)
    elif args.command:
        # Single command mode
        transcript = " ".join(args.command)
        router = MCPRouter()
        register_mock_handlers(router)
        process_transcript(agent, router, transcript)
    else:
        # Interactive mode
        router = MCPRouter()
        register_mock_handlers(router)
        print("\nInteractive mode. Type 'quit' or 'exit' to stop.")
        print("Type 'clear' to clear conversation history.\n")
        
        while True:
            try:
                transcript = input("You: ").strip()
                
                if not transcript:
                    continue
                    
                if transcript.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                    
                if transcript.lower() == "clear":
                    agent.clear_context()
                    print("[Conversation cleared]\n")
                    continue
                
                process_transcript(agent, router, transcript)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break


class ChatHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for chat API."""
    
    agent: Optional[ChatAgent] = None
    router: Optional[MCPRouter] = None
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                data = json.loads(body)
                message = data.get("message", "")
                session_id = data.get("session_id", "default")
                
                if not message:
                    self.send_error(400, "Missing 'message' field")
                    return
                
                # Process the message
                if self.agent is None:
                    self.send_error(500, "Agent not initialized")
                    return
                    
                response_data = process_chat(
                    self.agent,
                    self.router or MCPRouter(),
                    message,
                    session_id=session_id,
                )
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not found")
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def process_chat(
    agent: ChatAgent,
    router: MCPRouter,
    message: str,
    session_id: str = "default",
) -> dict[str, Any]:
    """Process chat message and return structured response."""
    try:
        # Use persistent event loop instead of asyncio.run()
        agent.set_session_id(session_id)
        response = run_async(agent.process_transcript(message))
        
        # Extract intent from response (format: "Recognized: INTENT (confidence: X.XX)")
        intent = "unknown"
        confidence = 0.0
        
        if "Recognized:" in response:
            # Parse: "Recognized: volume_control (confidence: 0.95)"
            parts = response.split("Recognized: ")[1]
            intent_part = parts.split(" (confidence:")[0]
            intent = intent_part.strip()
            
            # Extract confidence
            conf_part = parts.split("confidence: ")[1].rstrip(")")
            confidence = float(conf_part)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "tools_used": [],
            "session_id": session_id,
            "cache_stats": agent.get_cache_stats(),
        }
    except Exception as e:
        return {
            "intent": "error",
            "response": str(e),
            "tools_used": [],
            "session_id": session_id,
            "error": str(e)
        }


def start_http_server(agent: ChatAgent, port: int):
    """Start HTTP API server."""
    ChatHTTPHandler.agent = agent
    ChatHTTPHandler.router = MCPRouter()
    
    server = HTTPServer(("127.0.0.1", port), ChatHTTPHandler)
    print(f"\nChat API running on http://127.0.0.1:{port}")
    print("Endpoint: POST /chat")
    print("Request: {\"message\": \"your message here\", \"session_id\": \"default\"}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


def process_transcript(agent: ChatAgent, router: MCPRouter, transcript: str):
    """Process a single transcript and print results."""
    try:
        response = run_async(agent.process_transcript(transcript))
        
        print(f"\nJARVIS: {response}")
        print()
    except Exception as e:
        print(f"\nError: {str(e)}\n")


def format_result(tool_name: str, result: dict) -> str:
    """Format tool result into human-friendly response."""
    if tool_name == "get_system_info":
        cpu = result.get("cpu", "N/A")
        ram = result.get("ram", {})
        ram_pct = ram.get("percent", "N/A")
        ram_used = ram.get("used_gb", "N/A")
        ram_total = ram.get("total_gb", "N/A")
        network = result.get("network", {})
        connected = "connected" if network.get("connected") else "disconnected"
        interface = network.get("interface", "Unknown")
        
        return (
            f"Your CPU usage is {cpu}%. "
            f"RAM: {ram_used}GB / {ram_total}GB ({ram_pct}% used). "
            f"Network: {connected} via {interface}."
        )
    
    elif tool_name == "control_volume":
        if "level" in result:
            return f"Volume set to {result['level']}%."
        elif "muted" in result:
            return "Muted." if result["muted"] else "Unmuted."
        return "Volume adjusted."
    
    elif tool_name == "control_spotify":
        action = result.get("action", "")
        if action == "play":
            return "Playing music."
        elif action == "pause":
            return "Music paused."
        elif action == "next":
            return "Skipping to next track."
        elif action == "previous":
            return "Going to previous track."
        elif "track" in result:
            track = result["track"]
            return f"Now playing: {track.get('name', 'Unknown')} by {track.get('artist', 'Unknown')}."
        return "Spotify command executed."
    
    elif tool_name == "toggle_network":
        interface = result.get("interface", "Network")
        enabled = result.get("enabled", False)
        status = "enabled" if enabled else "disabled"
        return f"{interface} {status}."
    
    return str(result)


def register_mock_handlers(router: MCPRouter):
    """Register mock handlers for testing without MCP Server."""
    pass  # Router uses fallback direct execution



if __name__ == "__main__":
    main()
