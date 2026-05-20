"""
src/jarvis/ui/tools.py
──────────────────────
Tool listing and help display for JARVIS CLI.
"""
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .theme import run_async

console = Console()

SHOW_TOOLS_PATTERNS: frozenset[str] = frozenset({
    "show all tools", "show tools", "list tools", "list all tools",
    "what tools do you have", "what tools are available", "available tools",
    "show available tools", "what can you do", "what commands do you have",
    "show capabilities", "show all commands", "list commands",
})

# Explicit allowlist — anything NOT in this set goes to Spotify / Music
SYSTEM_TOOL_NAMES: frozenset[str] = frozenset({
    "get_system_info",
    "resolve_path",
    "list_directory",
    "read_file",
    "write_file",
    "run_command",
    "get_network_info",
    "get_battery_info",
})


def show_tools(agent) -> None:
    """Fetch all tools from MCP and display them in a formatted table."""
    from jarvis.spotify.handlers import get_mcp_router
    router = get_mcp_router(agent)
    if router is None:
        console.print("[bold red]Cannot reach MCP router.[/]\n")
        return

    console.print("\n [bold cyan]Fetching tools from MCP servers…[/]")
    try:
        raw_tools: list[dict] = run_async(router.list_tools())
    except Exception as exc:
        console.print(f"[bold red]Failed to fetch tools:[/] {exc}\n")
        return

    if not raw_tools:
        console.print("[yellow]No tools returned by MCP servers.[/]\n")
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    for t in raw_tools:
        name: str = t.get("name", "unknown")
        group = "System" if name in SYSTEM_TOOL_NAMES else "Spotify / Music"
        groups[group].append(t)

    table = Table(
        title=f"[bold white]JARVIS Available Tools ({len(raw_tools)} total)[/]",
        show_header=True, header_style="bold cyan",
        border_style="dim", expand=False, padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool Name", style="bold cyan", min_width=30)
    table.add_column("Description", style="white", max_width=60)

    idx = 1
    for group_name, tools in sorted(groups.items()):
        table.add_section()
        table.add_row("", f"[bold magenta]{group_name}[/]", "", end_section=False)
        for t in sorted(tools, key=lambda x: x.get("name", "")):
            raw_desc = t.get("description") or ""
            desc = raw_desc.split(".")[0].strip()[:55] + ("\u2026" if len(raw_desc.split(".")[0]) > 55 else "")
            table.add_row(str(idx), t.get("name", "unknown"), desc)
            idx += 1

    console.print()
    console.print(table)
    # Pick two real tool names for the tip — one from each category if possible
    system_eg = next((t.get("name", "") for t in raw_tools if t.get("name") in SYSTEM_TOOL_NAMES), "")
    spotify_eg = next((t.get("name", "") for t in raw_tools if t.get("name") not in SYSTEM_TOOL_NAMES), "")
    tip_parts = [f"[bold cyan]{n}[/]" for n in [system_eg, spotify_eg] if n]
    tip_examples = " or ".join(tip_parts) if tip_parts else "[bold cyan]get system info[/]"
    console.print(
        f"\n [dim]Tip: ask JARVIS to use any tool by name, e.g. {tip_examples}[/]\n"
    )


def show_help() -> None:
    help_text = """
 [bold cyan]? /help[/]       show full help                      [bold magenta]ctrl+a[/]    go to line start
 [bold cyan]/[/]           commands                            [bold magenta]ctrl+e[/]    go to line end
 [bold cyan]/tools[/]      list all available MCP tools         [bold magenta]ctrl+u[/]    delete to beginning
 [bold cyan]/model[/]      choose AI model (selector)           [bold magenta]ctrl+k[/]    delete to end
 [bold cyan]/models[/]     list all models (with details)       [bold magenta]ctrl+c[/]    cancel / exit
 [bold cyan]/theme[/]      switch themes
 [bold cyan]clear[/]       clear conversation
"""
    console.print(Panel(help_text, title="JARVIS Help", border_style="cyan"))
