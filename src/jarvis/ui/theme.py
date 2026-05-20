"""
src/jarvis/ui/theme.py
──────────────────────
Console, theme management, banner and prompt helpers.
"""
import asyncio
import threading
from importlib.metadata import version as _pkg_version, PackageNotFoundError
import questionary
from rich.console import Console

console = Console()

from pathlib import Path
import tomllib

def _load_version() -> str:
    try:
        pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return str(data.get("project", {}).get("version", "0.6.7"))
    except Exception:
        pass
    try:
        return _pkg_version("jarvis-chat")
    except PackageNotFoundError:
        pass
    return "0.6.7"

_APP_VERSION = _load_version()

# ── Event loop ───────────────────────────────────────────────────────────────────

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
        raise RuntimeError("Event loop is already running")
    return loop.run_until_complete(coro)


# ── Theme ─────────────────────────────────────────────────────────────────────

_current_theme = "default"


def get_current_theme() -> str:
    return _current_theme


def set_current_theme(theme: str) -> None:
    global _current_theme
    _current_theme = theme


def get_questionary_style() -> questionary.Style:
    if _current_theme == "dim":
        return questionary.Style([
            ('pointer', 'fg:#666666 bold'), ('selected', 'fg:#888888'),
            ('instruction', 'fg:#444444'),
        ])
    if _current_theme == "high-contrast":
        return questionary.Style([
            ('pointer', 'fg:#ffffff bold'), ('selected', 'fg:#ffffff bold'),
            ('instruction', 'fg:#aaaaaa'),
        ])
    if _current_theme == "colorblind":
        return questionary.Style([
            ('pointer', 'fg:#0000ff bold'), ('selected', 'fg:#0000ff'),
            ('instruction', 'fg:#555555'),
        ])
    return questionary.Style([
        ('pointer', 'fg:#00ffff bold'), ('selected', 'fg:#00ffff'),
        ('instruction', 'fg:#888888'),
    ])


# ── Terminal UI helpers ─────────────────────────────────────────────────────

def get_git_info() -> str:
    import subprocess
    try:
        b = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if b:
            return f" [⏷ {b}]"
    except Exception:
        pass
    return ""


def print_banner(agent=None) -> None:
    import shutil
    cols = min(shutil.get_terminal_size().columns, 100)
    mcp_count = 1
    if agent:
        try:
            mcp_count = agent.get_mcp_server_count()
        except Exception:
            pass
    banner = (
        f"╭{'─'*(cols-2)}╮\n"
        f"│  ╭─╮╭─╮{' '*(cols-11)}│\n"
        f"│  ╰─╯╰─╯  [bold white]JARVIS Chat Agent v{_APP_VERSION}[/]{' '*(cols-36)}│\n"
        f"│  █ ▘▝ █  Describe a task to get started.{' '*(cols-43)}│\n"
        f"│   └└┘┘{' '*(cols-10)}│\n"
        f"│  Tip: /theme View or set color mode{' '*(cols-39)}│\n"
        f"│  JARVIS uses AI. Check for mistakes.{' '*(cols-40)}│\n"
        f"╰{'─'*(cols-2)}╯\n"
        f"\n● Environment loaded: 1 provider, {mcp_count} MCP servers active"
        f"\n● JARVIS Skills Server: Connected\n"
    )
    console.print(banner)


def print_prompt_header(agent) -> None:
    import shutil
    import os
    cols = shutil.get_terminal_size().columns
    cwd = os.getcwd()
    console.print(f"\n [bold blue]{cwd}[/][bold cyan]{get_git_info()}[/]")
    console.print("[dim]" + "─" * cols + "[/]")


def print_prompt_footer(agent) -> None:
    import shutil
    cols = shutil.get_terminal_size().columns
    console.print("[dim]" + "─" * cols + "[/]")
    left = " / commands · ? help"
    right = f"{agent.llm_config.provider}/{agent.llm_config.model}"
    padding = max(1, cols - len(left) - len(right) - 1)
    console.print(f"[dim]{left}{' ' * padding}[/][bold blue]{right}[/]")
