"""
src/jarvis/ui/models.py
───────────────────────
Model selector and listing UI for JARVIS CLI.
"""
import questionary
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.keys import Keys
from rich.console import Console

from .theme import get_questionary_style

console = Console()

KNOWN_COSTS: dict[str, str] = {
    "flash": "0.1x", "pro": "1.0x", "lite": "0.05x",
    "gpt-4o-mini": "0.1x", "gpt-4o": "1.0x", "o1": "10x", "o3-mini": "0.5x",
}


def is_core_model(m: str) -> bool:
    """Filter out irrelevant or obscure models."""
    ml = m.lower()
    if any(x in ml for x in ["tts", "vision", "image", "robotics", "banana", "lyria",
                               "embedding", "bison", "learnlm"]):
        return False
    if any(x in ml for x in ["gemini-1.5", "gemini-2.", "gemini-3.", "gemma-4",
                               "gpt-4", "gpt-o"]):
        return True
    if "preview" in ml and not ("pro" in ml or "flash" in ml):
        return False
    return True


def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def _cost_str(model: str, provider: str) -> str:
    if provider == "ollama":
        return "Local"
    ml = model.lower()
    if "gemma" in ml or "llama" in ml:
        return "via API Key"
    for k, v in KNOWN_COSTS.items():
        if k in ml:
            return v
    return "1x"


def _make_esc_binding(q):
    kb = KeyBindings()

    @kb.add(Keys.Escape)
    def _(event):
        event.app.exit(result=None)

    bindings: list[KeyBindingsBase] = [kb]
    if isinstance(q.application.key_bindings, KeyBindingsBase):
        bindings.insert(0, q.application.key_bindings)
    q.application.key_bindings = merge_key_bindings(bindings)


def list_all_models(agent) -> None:
    """Print all available models with token limits and cost info."""
    console.print("\n [bold white]Available AI Models (Detailed List):[/]\n")
    try:
        models = agent.get_available_models_detailed()
        provider = agent.llm_config.provider
        
        for m in models:
            name = m.get("name", "N/A")
            token_str = format_tokens(m.get("input_token_limit", 0))
            cost = _cost_str(name, provider)
            
            # Highlight active model
            if name == agent.llm_config.model:
                star = "[bold yellow]*[/] "
                name_style = "bold cyan"
            else:
                star = "  "
                name_style = "cyan"
                
            display_name = f"{name} ({provider})"
            label = f"{star}[{name_style}]{display_name:<45}[/]"
            console.print(f"{label} [dim]Context:[/] [green]{token_str:>6}[/] [dim]•[/] [bold]{cost:>5}[/]")
            
        console.print("\n [dim]Use [bold cyan]/model <name>[/] to switch directly or [bold cyan]/model[/] for the selector.[/]\n")
    except Exception as e:
        console.print(f"[bold red]Error fetching models:[/] {e}")


def handle_model_switch(agent) -> None:
    """Interactive model selector using questionary."""
    from chat_agent.llm import create_provider

    console.print("Fetching available models...", style="dim")
    raw_choices: list[tuple[str, str, int]] = []

    try:
        raw_choices.extend(
            (m["name"], agent.llm_config.provider, m.get("input_token_limit", 0))
            for m in agent.get_available_models_detailed()
            if is_core_model(m["name"])
        )
    except Exception:
        raw_choices.append((agent.llm_config.model, agent.llm_config.provider, 0))

    for p in ["gemini", "openai", "ollama"]:
        if p == agent.llm_config.provider:
            continue
        try:
            temp = create_provider(p)
            if not temp.is_configured():
                continue
            if hasattr(temp, "get_available_models_detailed"):
                raw_choices.extend(
                    (m["name"], p, m.get("input_token_limit", 0))
                    for m in temp.get_available_models_detailed()
                    if is_core_model(m["name"])
                )
            else:
                raw_choices.extend(
                    (m, p, 0) for m in temp.get_available_models() if is_core_model(m)
                )
        except Exception:
            pass

    raw_choices.sort(key=lambda x: (x[0] != agent.llm_config.model, x[1], x[0]))
    seen: set[tuple[str, str]] = set()
    choices = []
    console.print("\nChoose the AI model to use for JARVIS.\n", style="bold")

    for model, provider, tokens in raw_choices:
        if (model, provider) in seen:
            continue
        seen.add((model, provider))
        cost = _cost_str(model, provider)
        tok = format_tokens(tokens) if tokens > 0 else "N/A"
        label = f"* {model} ({provider})" if model == agent.llm_config.model else f"  {model} ({provider})"
        pad = max(1, 55 - len(label))
        title = f"{label}{' ' * pad}[dim]Context:[/] [green]{tok:>6}[/] [dim]•[/] [bold]{cost:>5}[/]"
        choices.append(questionary.Choice(title=title, value=(model, provider)))

    q = questionary.select(
        "", choices=choices, style=get_questionary_style(),
        pointer="❯", qmark="",
        instruction="\n↑↓ to navigate · Enter to select · Esc to cancel\n",
    )
    _make_esc_binding(q)
    selection = q.ask()

    if selection:
        model_name, provider_name = selection
        try:
            agent.change_model(model_name, provider_name)
            prov_msg = f" (via {provider_name})" if provider_name else ""
            console.print(f"[bold green]✓ Model changed to {model_name}{prov_msg}[/]\n")
        except Exception as e:
            console.print(f"[bold red]Error changing model: {e}[/]\n")
