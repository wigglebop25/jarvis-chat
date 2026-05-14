import sys
import os
import argparse
import json
import asyncio
import threading
import questionary
from typing import Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chat_agent import ChatAgent, AgentConfig

console = Console()

# Global event loop for persistent async operations
_event_loop = None
_loop_lock = threading.Lock()

# Global theme setting
_current_theme = "default"


def get_git_info():
    import subprocess
    try:
        b = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True).stdout.strip()
        if b:
            return f" [⎇ {b}]"
    except Exception:
        pass
    return ""

def print_banner(agent=None):
    import shutil
    cols = shutil.get_terminal_size().columns
    if cols > 100:
        cols = 100
    
    mcp_count = 1
    if agent:
        try:
            mcp_count = agent.get_mcp_server_count()
        except Exception:
            mcp_count = 1
    
    banner = f"""╭{'─'*(cols-2)}╮
│  ╭─╮╭─╮{' '*(cols-11)}│
│  ╰─╯╰─╯  [bold white]JARVIS Chat Agent v0.1.0[/]{' '*(cols-36)}│
│  █ ▘▝ █  Describe a task to get started.{' '*(cols-43)}│
│   ▔▔▔▔{' '*(cols-10)}│
│  Tip: /theme View or set color mode{' '*(cols-39)}│
│  JARVIS uses AI. Check for mistakes.{' '*(cols-40)}│
╰{'─'*(cols-2)}╯

● Environment loaded: 1 provider, {mcp_count} MCP servers active
● JARVIS Skills Server: Connected
"""
    console.print(banner)

def print_prompt_header(agent):
    import shutil
    import os
    cols = shutil.get_terminal_size().columns
    cwd = os.getcwd()
    git_info = get_git_info()
    
    console.print(f"\n [bold blue]{cwd}[/][bold cyan]{git_info}[/]")
    console.print("[dim]" + "─" * cols + "[/]")
    
def print_prompt_footer(agent):
    import shutil
    cols = shutil.get_terminal_size().columns
    
    console.print("[dim]" + "─" * cols + "[/]")
    
    left = " / commands · ? help"
    right = f"{agent.llm_config.provider}/{agent.llm_config.model}"
    padding = max(1, cols - len(left) - len(right) - 1)
    
    console.print(f"[dim]{left}{' ' * padding}[/][bold blue]{right}[/]")

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
        raise RuntimeError("Event loop is already running")
    return loop.run_until_complete(coro)

def get_questionary_style():
    if _current_theme == "dim":
        return questionary.Style([('pointer', 'fg:#666666 bold'), ('selected', 'fg:#888888'), ('instruction', 'fg:#444444')])
    elif _current_theme == "high-contrast":
        return questionary.Style([('pointer', 'fg:#ffffff bold'), ('selected', 'fg:#ffffff bold'), ('instruction', 'fg:#aaaaaa')])
    elif _current_theme == "colorblind":
        return questionary.Style([('pointer', 'fg:#0000ff bold'), ('selected', 'fg:#0000ff'), ('instruction', 'fg:#555555')])
    else:
        return questionary.Style([('pointer', 'fg:#00ffff bold'), ('selected', 'fg:#00ffff'), ('instruction', 'fg:#888888')])

def show_help():
    help_text = """
 [bold cyan]? /help[/]       show full help                      [bold magenta]ctrl+a[/]    go to line start
 [bold cyan]/[/]           commands                            [bold magenta]ctrl+e[/]    go to line end
 [bold cyan]/model[/]      choose AI model (selector)           [bold magenta]ctrl+u[/]    delete to beginning
 [bold cyan]/models[/]     list all models (with details)       [bold magenta]ctrl+k[/]    delete to end
 [bold cyan]/theme[/]      switch themes                       [bold magenta]ctrl+c[/]    cancel / exit
 [bold cyan]clear[/]       clear conversation
"""
    console.print(Panel(help_text, title="JARVIS Help", border_style="cyan"))

def is_core_model(m):
    """Filter out weird obscure models unless it's a common one."""
    m_lower = m.lower()
    if "tts" in m_lower or "vision" in m_lower or "image" in m_lower or "robotics" in m_lower:
        return False
    if "banana" in m_lower or "lyria" in m_lower: # Keep deep-research if user wants it
        return False
    if "embedding" in m_lower or "bison" in m_lower or "learnlm" in m_lower:
        return False
    # Allow all Gemini 2.x, 3.x, and Gemma 4
    if any(x in m_lower for x in ["gemini-1.5", "gemini-2.", "gemini-3.", "gemma-4", "gpt-4", "gpt-o"]):
        return True
    if "preview" in m_lower and not ("pro" in m_lower or "flash" in m_lower):
        return False
    return True

def format_tokens(n):
    if n >= 1000000:
        return f"{n/1000000:.1f}M"
    if n >= 1000:
        return f"{n/1000:.0f}K"
    return str(n)

def list_all_models(agent):
    console.print("\n [bold white]Available AI Models (Detailed List):[/]\n")
    
    try:
        detailed_models = agent.get_available_models_detailed()
        
        for m in detailed_models:
            name = m.get("name", "N/A")
            tokens = m.get("input_token_limit", 0)
            token_str = format_tokens(tokens)
            
            is_active = name == agent.llm_config.model
            star = "[bold yellow]*[/] " if is_active else "  "
            
            console.print(f"{star}[bold cyan]{name:<35}[/] [dim]Context:[/] [green]{token_str:>6}[/]")
            
        console.print("\n [dim]Use [bold cyan]/model <name>[/] to switch directly or [bold cyan]/model[/] for the selector.[/]\n")
        
    except Exception as e:
        console.print(f"[bold red]Error fetching models:[/] {e}")

def handle_model_switch(agent):
    from chat_agent.llm import create_provider
    
    KNOWN_COSTS = {
        "flash": "0.1x",
        "pro": "1.0x",
        "lite": "0.05x",
        "gpt-4o-mini": "0.1x",
        "gpt-4o": "1.0x",
        "o1": "10x",
        "o3-mini": "0.5x"
    }

    def get_cost_str(m, p):
        if p == "ollama":
            return "Free (Local)"
        m_lower = m.lower()
        if "gemma" in m_lower or "llama" in m_lower:
            return "via API Key"
        for k, v in KNOWN_COSTS.items():
            if k in m_lower:
                return v
        return "1x"
    
    console.print("Fetching available models...", style="dim")
    raw_choices = []
    
    try:
        current_models = agent.get_available_models_detailed()
        raw_choices.extend([(m["name"], agent.llm_config.provider, m.get("input_token_limit", 0)) for m in current_models if is_core_model(m["name"])])
    except Exception:
        raw_choices.append((agent.llm_config.model, agent.llm_config.provider, 0))
        
    for p in ["gemini", "openai", "ollama"]:
        if p != agent.llm_config.provider:
            try:
                temp_p = create_provider(p)
                if temp_p.is_configured():
                    if hasattr(temp_p, "get_available_models_detailed"):
                        other_models = temp_p.get_available_models_detailed()
                        raw_choices.extend([(m["name"], p, m.get("input_token_limit", 0)) for m in other_models if is_core_model(m["name"])])
                    else:
                        other_models = temp_p.get_available_models()
                        raw_choices.extend([(m, p, 0) for m in other_models if is_core_model(m)])
            except Exception:
                pass
                
    seen = set()
    unique_choices = []
    
    # Sort: put current model first, then others by name
    raw_choices.sort(key=lambda x: (x[0] != agent.llm_config.model, x[1], x[0]))

    console.print("\nChoose the AI model to use for JARVIS.\n", style="bold")
    
    for m, p, t in raw_choices:
        if (m, p) not in seen:
            seen.add((m, p))
            cost = get_cost_str(m, p)
            token_str = format_tokens(t) if t > 0 else "N/A"
            
            # Format nicely
            display_name = f"{m} ({p})"
            if m == agent.llm_config.model:
                display_name = f"[bold yellow]* {display_name}[/]"
            else:
                display_name = f"  {display_name}"
            
            # Align right
            padding = max(1, 55 - len(display_name))
            title = f"{display_name}{' ' * padding}[dim]Context:[/] [green]{token_str:>6}[/] [dim]•[/] [bold]{cost:>5}[/]"
            unique_choices.append(questionary.Choice(title=title, value=(m, p)))
    
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.keys import Keys
    
    q = questionary.select(
        "",
        choices=unique_choices,
        style=get_questionary_style(),
        pointer="❯",
        qmark="",
        instruction="\n↑↓ to navigate · Enter to select · Esc to cancel\n"
    )
    
    kb = KeyBindings()
    @kb.add(Keys.Escape)
    def _(event):
        event.app.exit(result=None)
        
    from prompt_toolkit.key_binding import KeyBindingsBase
    
    # Ensure bindings is a Sequence of KeyBindingsBase
    bindings: list[KeyBindingsBase] = [kb]
    if isinstance(q.application.key_bindings, KeyBindingsBase):
        bindings.insert(0, q.application.key_bindings)
        
    q.application.key_bindings = merge_key_bindings(bindings)
    
    selection = q.ask()
    
    if selection:
        model_name, provider_name = selection
        try:
            agent.change_model(model_name, provider_name)
            prov_msg = f" (via {provider_name})" if provider_name else ""
            console.print(f"[bold green]✓ Model changed to {model_name}{prov_msg}[/]\n")
        except Exception as e:
            console.print(f"[bold red]Error changing model: {e}[/]\n")

def main():
    global _current_theme
    parser = argparse.ArgumentParser(description="JARVIS Chat Agent CLI")
    parser.add_argument("--server", "-s", action="store_true", help="Start HTTP API server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--mcp-url", default="http://127.0.0.1:5050", help="MCP server URL")
    parser.add_argument("command", nargs="*", help="Command to execute")
    
    args = parser.parse_args()
    os.environ["MCP_SERVER_URL"] = args.mcp_url
    
    agent = ChatAgent(AgentConfig())
    
    if args.server:
        start_http_server(agent, args.port)
    elif args.command:
        transcript = " ".join(args.command)
        process_transcript(agent, transcript)
    else:
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

                if transcript.lower() == "/help":
                    show_help()
                    continue

                if transcript.lower() == "/models":
                    list_all_models(agent)
                    continue

                if transcript.lower().startswith("/theme"):
                    parts = transcript.split()
                    if len(parts) > 1 and parts[1] in ["default", "dim", "high-contrast", "colorblind"]:
                        _current_theme = parts[1]
                        console.print(f"[bold green]✓ Theme set to {_current_theme}[/]\n")
                    else:
                        import questionary
                        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
                        from prompt_toolkit.keys import Keys
                        
                        theme_choices = [
                            questionary.Choice(title="default", value="default"),
                            questionary.Choice(title="dim", value="dim"),
                            questionary.Choice(title="high-contrast", value="high-contrast"),
                            questionary.Choice(title="colorblind", value="colorblind"),
                            questionary.Choice(title="Cancel", value=None)
                        ]
                        
                        q = questionary.select(
                            "Select a Theme:",
                            choices=theme_choices,
                            style=get_questionary_style(),
                            pointer="❯",
                            qmark=""
                        )
                        
                        kb = KeyBindings()
                        @kb.add(Keys.Escape)
                        def _(event):
                            event.app.exit(result=None)
                            
                        from prompt_toolkit.key_binding import KeyBindingsBase
                        bindings: list[KeyBindingsBase] = [kb]
                        if isinstance(q.application.key_bindings, KeyBindingsBase):
                            bindings.insert(0, q.application.key_bindings)
                            
                        q.application.key_bindings = merge_key_bindings(bindings)
                        
                        selection = q.ask()
                        if selection:
                            _current_theme = selection
                            console.print(f"[bold green]✓ Theme set to {_current_theme}[/]\n")
                    continue
                
                if transcript.lower().startswith("/model"):
                    parts = transcript.split(maxsplit=2)
                    if len(parts) >= 2:
                        model_name = parts[1]
                        provider_name = parts[2] if len(parts) > 2 else None
                        try:
                            agent.change_model(model_name, provider_name)
                            prov_msg = f" (via {provider_name})" if provider_name else ""
                            console.print(f"[bold green]✓ Model changed to {model_name}{prov_msg}[/]\n")
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


class ChatHTTPHandler(BaseHTTPRequestHandler):
    agent: Optional[ChatAgent] = None
    
    def do_POST(self):
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
                if self.agent is None:
                    self.send_error(500, "Agent not initialized")
                    return
                response_data = process_chat(self.agent, message, session_id=session_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not found")
    
    def log_message(self, format, *args):
        pass


def process_chat(agent: ChatAgent, message: str, session_id: str = "default") -> dict[str, Any]:
    try:
        agent.set_session_id(session_id)
        response = run_async(agent.process_transcript(message))
        intent = "unknown"
        confidence = 0.0
        if "Recognized:" in response:
            parts = response.split("Recognized: ")[1]
            intent = parts.split(" (confidence:")[0].strip()
            confidence = float(parts.split("confidence: ")[1].rstrip(")"))
        return {
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "tools_used": [],
            "session_id": session_id,
            "cache_stats": agent.get_cache_stats(),
        }
    except Exception as e:
        return {"intent": "error", "response": str(e), "tools_used": [], "session_id": session_id, "error": str(e)}


def start_http_server(agent: ChatAgent, port: int):
    ChatHTTPHandler.agent = agent
    server = HTTPServer(("127.0.0.1", port), ChatHTTPHandler)
    print(f"\nChat API running on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def process_transcript(agent: ChatAgent, transcript: str):
    try:
        # Reset last_usage
        if hasattr(agent, 'last_usage'):
            agent.last_usage = {}
            
        response = run_async(agent.process_transcript(transcript))
        
        console.print(f"\n[bold magenta]JARVIS:[/] {response}")
        
        # Display token usage like Copilot CLI
        if hasattr(agent, 'last_usage') and agent.last_usage:
            u = agent.last_usage
            if u.get("cache_hit"):
                console.print(Text("Answered from local cache (0 tokens)", style="dim italic"))
            elif u.get("prompt_tokens") == 0 and u.get("completion_tokens") == 0:
                console.print(Text("Executed via Fast Path (0 tokens)", style="dim italic"))
            else:
                prompt_tokens = u.get("prompt_tokens", 0)
                comp_tokens = u.get("completion_tokens", 0)
                if prompt_tokens or comp_tokens:
                    console.print(Text(f"Usage: {prompt_tokens} prompt • {comp_tokens} completion", style="dim italic"))
        print()
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/]\n")


if __name__ == "__main__":
    main()
