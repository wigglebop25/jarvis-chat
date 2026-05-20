"""
src/jarvis/spotify/phrases.py
─────────────────────────────
Phrase sets for the Spotify fast-path dispatcher.

Each frozenset covers the natural-language variations of one deterministic
command. To add a new voice variation: add one string to the right set.
To add a brand-new command: add a new frozenset and a tuple to EXACT_DISPATCH.
"""

# ── Per-command phrase sets ───────────────────────────────────────────────────

NOW_PLAYING_PHRASES: frozenset[str] = frozenset({
    "show what song is playing", "what song is playing", "what's playing",
    "whats playing", "show now playing", "now playing", "what is playing",
    "what song is this", "current song", "current track", "show song",
})

QUEUE_PHRASES: frozenset[str] = frozenset({
    "show queue", "queue", "show my queue", "my queue",
    "what is in queue", "what's in queue", "whats in queue",
    "show the queue", "upcoming tracks", "upcoming songs",
})

PAUSE_PHRASES: frozenset[str] = frozenset({
    "pause", "pause music", "pause playback", "pause song", "pause track",
    "stop music", "stop playback", "stop playing", "stop the music",
})

RESUME_PHRASES: frozenset[str] = frozenset({
    "resume", "resume music", "resume playback", "resume playing",
    "unpause", "unpause music", "continue", "continue playing",
    "continue music", "start playback",
})

SKIP_NEXT_PHRASES: frozenset[str] = frozenset({
    "next", "next song", "next track", "skip", "skip song", "skip track",
    "skip to next", "play next", "next please",
})

SKIP_PREVIOUS_PHRASES: frozenset[str] = frozenset({
    "previous", "previous song", "previous track", "go back", "back",
    "last song", "last track", "skip to previous", "play previous",
})

MY_PLAYLISTS_PHRASES: frozenset[str] = frozenset({
    "show my playlists", "show my playlist", "list my playlists",
    "spotify playlists", "my playlists", "get my playlists",
    "show playlists", "list playlists",
})

DEVICES_PHRASES: frozenset[str] = frozenset({
    "show devices", "show available devices", "list devices",
    "available devices", "spotify devices", "what devices",
    "my devices", "show spotify devices",
})

RECENTLY_PLAYED_PHRASES: frozenset[str] = frozenset({
    "recently played", "show recently played", "recent tracks",
    "recent songs", "listening history", "show history",
    "what i played recently", "what have i been listening to",
    "show recent tracks", "show recent songs",
})

LIKED_SONGS_PHRASES: frozenset[str] = frozenset({
    "liked songs", "show liked songs", "my liked songs",
    "saved tracks", "show saved tracks", "my saved tracks",
    "favorites", "my favorites", "show favorites",
    "get liked songs", "get saved tracks",
})

# ── Dispatch table ────────────────────────────────────────────────────────────
# Maps (phrase_set, tool_name, kwargs). The dispatcher iterates in order.
# NOTE: QUEUE_PHRASES is intentionally absent — handled by handle_show_queue()
# which uses playlist track data for accuracy instead of the getQueue API.

EXACT_DISPATCH: tuple[tuple[frozenset[str], str, dict], ...] = (
    (NOW_PLAYING_PHRASES,     "getNowPlaying",       {}),
    (PAUSE_PHRASES,           "pausePlayback",       {}),
    (RESUME_PHRASES,          "resumePlayback",      {}),
    (SKIP_NEXT_PHRASES,       "skipToNext",          {}),
    (SKIP_PREVIOUS_PHRASES,   "skipToPrevious",      {}),
    (MY_PLAYLISTS_PHRASES,    "getMyPlaylists",      {"limit": 50}),
    (DEVICES_PHRASES,         "getAvailableDevices", {}),
    (RECENTLY_PLAYED_PHRASES, "getRecentlyPlayed",   {"limit": 20}),
    (LIKED_SONGS_PHRASES,     "getUsersSavedTracks", {"limit": 50}),
)
