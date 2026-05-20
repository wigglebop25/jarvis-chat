"""
main.py — JARVIS Chat Agent entry point.

Responsibilities (and ONLY these):
  - CLI argument parsing
  - REPL loop + slash command routing
  - process_transcript() — per-message coordinator
  - HTTP API server

Everything else lives in src/jarvis/.
"""

import sys
import os
import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chat_agent import ChatAgent, AgentConfig

from jarvis.ui.theme import console, run_async, print_banner, print_prompt_header, print_prompt_footer, get_questionary_style, set_current_theme
from jarvis.ui.tools import show_tools, show_help, SHOW_TOOLS_PATTERNS
from jarvis.ui.models import handle_model_switch, list_all_models
from jarvis.spotify.dispatcher import try_spotify_fast_path
from jarvis.spotify.guard import confirm_destructive_action
from jarvis.spotify.playlist_ui import maybe_expand_playlist_request

from rich.text import Text


# ── Per-message coordinator ───────────────────────────────────────────────────

def process_transcript(agent: ChatAgent, transcript: str) -> None:
    try:
        transcript = maybe_expand_playlist_request(transcript, agent)

        # Meta-questions: list tools without calling the LLM
        if " ".join(transcript.lower().strip().split()) in SHOW_TOOLS_PATTERNS:
            show_tools(agent)
            return

        # Spotify fast-path: handle deterministic commands without the LLM
        fast_path_response = run_async(try_spotify_fast_path(agent, transcript))
        if fast_path_response:
            console.print(f"\n[bold magenta]JARVIS:[/] {fast_path_response}\n")
            return

        # Destructive-operation guard: confirm before sending to LLM
        if not confirm_destructive_action(transcript, get_questionary_style):
            return

        # Reset usage tracker then invoke the LangGraph agent
        if hasattr(agent, "last_usage"):
            agent.last_usage = {}

        response = run_async(agent.process_transcript(transcript))
        console.print(f"\n[bold magenta]JARVIS:[/] {response}")

        # Token usage display
        if hasattr(agent, "last_usage") and agent.last_usage:
            u = agent.last_usage
            if u.get("cache_hit"):
                console.print(Text("Answered from local cache (0 tokens)", style="dim italic"))
            elif u.get("prompt_tokens") == 0 and u.get("completion_tokens") == 0:
                console.print(Text("Executed via Fast Path (0 tokens)", style="dim italic"))
            else:
                p, c = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                if p or c:
                    console.print(Text(f"Usage: {p} prompt • {c} completion", style="dim italic"))
        print()
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/]\n")


# ── REPL loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    import questionary
    from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
    from prompt_toolkit.keys import Keys

    parser = argparse.ArgumentParser(description="JARVIS Chat Agent CLI")
    parser.add_argument("--server", "-s", action="store_true", help="Start HTTP API server")
    parser.add_argument("--port",   "-p", type=int, default=8000)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:5050")
    parser.add_argument("command", nargs="*")
    args = parser.parse_args()
    os.environ["MCP_SERVER_URL"] = args.mcp_url

    agent = ChatAgent(AgentConfig())

    if args.server:
        start_http_server(agent, args.port)
        return
    if args.command:
        process_transcript(agent, " ".join(args.command))
        return

    print_banner(agent)

    while True:
        try:
            print_prompt_header(agent)
            transcript = input("❯ ").strip()
            print_prompt_footer(agent)

            if not transcript:
                continue
            if transcript.lower() in ("quit", "exit", "q"):
                console.print("Goodbye!")
                break
            if transcript.lower() == "clear":
                agent.clear_context()
                console.print("[bold green]✓ Conversation cleared[/]\n")
                continue
            if transcript.lower() in ("/help", "?"):
                show_help()
                continue
            if transcript.lower() == "/tools":
                show_tools(agent)
                continue
            if transcript.lower() == "/models":
                list_all_models(agent)
                continue

            if transcript.lower().startswith("/theme"):
                parts = transcript.split()
                valid = ["default", "dim", "high-contrast", "colorblind"]
                if len(parts) > 1 and parts[1] in valid:
                    set_current_theme(parts[1])
                    console.print(f"[bold green]✓ Theme set to {parts[1]}[/]\n")
                else:
                    choices = [questionary.Choice(title=t, value=t) for t in valid]
                    choices.append(questionary.Choice(title="Cancel", value=None))
                    q = questionary.select("Select a Theme:", choices=choices,
                                           style=get_questionary_style(), pointer="❯", qmark="")
                    kb = KeyBindings()
                    @kb.add(Keys.Escape)
                    def _(event): event.app.exit(result=None)
                    bindings: list[KeyBindingsBase] = [kb]
                    if isinstance(q.application.key_bindings, KeyBindingsBase):
                        bindings.insert(0, q.application.key_bindings)
                    q.application.key_bindings = merge_key_bindings(bindings)
                    sel = q.ask()
                    if sel:
                        set_current_theme(sel)
                        console.print(f"[bold green]✓ Theme set to {sel}[/]\n")
                continue

            if transcript.lower().startswith("/model"):
                parts = transcript.split(maxsplit=2)
                if len(parts) >= 2:
                    try:
                        agent.change_model(parts[1], parts[2] if len(parts) > 2 else None)
                        console.print(f"[bold green]✓ Model changed to {parts[1]}[/]\n")
                    except Exception as e:
                        console.print(f"[bold red]Error changing model: {e}[/]\n")
                else:
                    handle_model_switch(agent)
                continue

            process_transcript(agent, transcript)

        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break
        except EOFError:
            break


# ── HTTP API server ───────────────────────────────────────────────────────────

class ChatHTTPHandler(BaseHTTPRequestHandler):
    agent: Any = None

    def do_POST(self) -> None:
        if self.path != "/chat":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode())
            message = data.get("message", "")
            session_id = data.get("session_id", "default")
            if not message:
                self.send_error(400, "Missing 'message' field")
                return
            if self.agent is None:
                self.send_error(500, "Agent not initialized")
                return
            resp = _process_chat(self.agent, message, session_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, fmt, *args) -> None:
        pass


def _process_chat(agent: ChatAgent, message: str, session_id: str = "default") -> dict:
    try:
        agent.set_session_id(session_id)
        response = run_async(agent.process_transcript(message))
        return {
            "response": response, "session_id": session_id,
            "cache_stats": agent.get_cache_stats(),
        }
    except Exception as e:
        return {"response": str(e), "session_id": session_id, "error": str(e)}


def start_http_server(agent: ChatAgent, port: int) -> None:
    ChatHTTPHandler.agent = agent
    server = HTTPServer(("127.0.0.1", port), ChatHTTPHandler)
    print(f"\nChat API running on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
