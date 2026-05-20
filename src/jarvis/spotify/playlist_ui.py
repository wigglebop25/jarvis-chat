"""
src/jarvis/spotify/playlist_ui.py
──────────────────────────────────
Interactive playlist disambiguation UI.

When the user types a generic phrase like "show my playlist", this module
presents a questionary menu to clarify intent, then optionally shows a
playlist picker so the user can select by number instead of typing a name.
"""

import re

import questionary

from jarvis.ui.theme import console, get_questionary_style, run_async
from .handlers import get_mcp_router, extract_mcp_text


# Generic phrases that trigger the disambiguation menu
_GENERIC_PLAYLIST_PROMPTS: frozenset[str] = frozenset({
    "playlist", "playlists",
    "my playlist", "my playlists",
    "show my playlist", "show playlist",
    "spotify playlist", "show spotify playlist",
})

_PLAYLIST_LINE_RE = re.compile(
    r'^\d+\.\s+"(.*?)"\s+\(\d+\s+tracks?\)\s+-\s+ID:\s*(\S+)$'
)


def _fetch_playlists(agent) -> list[dict]:
    """
    Synchronously fetch the user's Spotify playlists for use in selectors.
    Returns a list of {"name": str, "id": str} dicts.
    """
    try:
        router = get_mcp_router(agent)
        if router is None:
            return []
        result = run_async(router.execute_tool("getMyPlaylists", limit=50))
        if result.is_error:
            return []
        raw = extract_mcp_text(result.result) if not isinstance(result.result, str) else (result.result or "")
        playlists: list[dict] = []
        for line in raw.splitlines():
            m = _PLAYLIST_LINE_RE.match(line.strip())
            if m:
                playlists.append({"name": m.group(1), "id": m.group(2)})
        return playlists
    except Exception:
        return []


def maybe_expand_playlist_request(transcript: str, agent=None) -> str:
    """
    Copilot-style disambiguation for generic Spotify playlist requests.

    If the transcript is a generic playlist phrase, shows an interactive menu.
    For 'play' intent, fetches real playlists and shows a selector.
    Otherwise returns transcript unchanged.
    """
    normalized = " ".join(transcript.lower().strip().split())
    if normalized not in _GENERIC_PLAYLIST_PROMPTS:
        return transcript

    console.print("● Asked user What would you like to do with your Spotify playlist?")
    selection = questionary.select(
        "What would you like to do with your Spotify playlist?",
        choices=[
            questionary.Choice(title="Show my playlists",          value="show"),
            questionary.Choice(title="Search playlists",           value="search"),
            questionary.Choice(title="Play a playlist",            value="play"),
            questionary.Choice(title="Other (type your answer)",   value="other"),
        ],
        style=get_questionary_style(),
        pointer="❯", qmark="",
        instruction="\n↑↓ to select · Enter to confirm · Esc to cancel\n",
    ).ask()

    if selection == "show":
        return "show my playlists"

    if selection == "search":
        q = questionary.text(
            "Which playlist should I search for?",
            style=get_questionary_style(), qmark="",
        ).ask()
        return f"search playlist {q.strip()}" if q and q.strip() else "search playlists"

    if selection == "play":
        playlists = _fetch_playlists(agent) if agent else []
        if playlists:
            console.print("\n [bold cyan]Your Spotify Playlists:[/]")
            choices = [
                questionary.Choice(title=f"  {i + 1}. {p['name']}", value=p["name"])
                for i, p in enumerate(playlists)
            ]
            choices.append(questionary.Choice(title="  ✎  Type a name manually", value="__manual__"))
            picked = questionary.select(
                "Which playlist should I play?",
                choices=choices, style=get_questionary_style(),
                pointer="❯", qmark="",
                instruction="\n↑↓ to navigate · Enter to select\n",
            ).ask()
            if picked and picked != "__manual__":
                return f"play playlist {picked}"
        q = questionary.text(
            "Which playlist should I play?",
            style=get_questionary_style(), qmark="",
        ).ask()
        return f"play playlist {q.strip()}" if q and q.strip() else "play playlist"

    if selection == "other":
        q = questionary.text(
            "Tell me exactly what you want to do with playlists:",
            style=get_questionary_style(), qmark="",
        ).ask()
        return q.strip() if q and q.strip() else transcript

    return transcript
