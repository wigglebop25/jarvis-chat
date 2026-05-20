"""
src/jarvis/spotify/guard.py
───────────────────────────
Destructive-operation guard for Spotify commands.

Detects user intent to perform irreversible operations (remove tracks,
update/rename playlists) and asks for explicit confirmation before the
request is forwarded to the LLM and MCP server.
"""
import re
import questionary
from rich.console import Console

console = Console()

# ── Patterns that flag a destructive operation ──────────────────────────────

DESTRUCTIVE_PATTERNS: list[re.Pattern[str]] = [
    # Remove tracks from liked songs / library
    re.compile(
        r"\b(remove|delete|unlike|unfavorite|unsave)\b.{0,40}\b(song|track|music|it)\b",
        re.IGNORECASE,
    ),
    # Remove tracks from a playlist
    re.compile(
        r"\b(remove|delete)\b.{0,40}\b(from|off)\b.{0,40}\bplaylist\b",
        re.IGNORECASE,
    ),
    # Update / rename a playlist
    re.compile(
        r"\b(update|rename|edit|change)\b.{0,40}\bplaylist\b",
        re.IGNORECASE,
    ),
    # Clear / wipe an entire playlist
    re.compile(
        r"\b(clear|empty|wipe)\b.{0,40}\bplaylist\b",
        re.IGNORECASE,
    ),
]


def confirm_destructive_action(transcript: str, get_style_fn) -> bool:
    """
    Return True if the user confirms a destructive operation, False to cancel.

    Shows an inline questionary confirm prompt so the user must explicitly
    acknowledge before any irreversible Spotify change is sent to the LLM.
    Pass a zero-argument callable that returns a questionary Style object.
    """
    if not any(p.search(transcript) for p in DESTRUCTIVE_PATTERNS):
        return True  # Not destructive — proceed normally

    console.print(
        "\n [bold yellow]\u26a0 This action may permanently change your Spotify library.[/]"
    )
    confirmed = questionary.confirm(
        f'  Are you sure you want to: "{transcript.strip()}"?',
        default=False,
        style=get_style_fn(),
        qmark="",
    ).ask()

    if not confirmed:
        console.print(" [dim]Action cancelled.[/]\n")
    return bool(confirmed)
