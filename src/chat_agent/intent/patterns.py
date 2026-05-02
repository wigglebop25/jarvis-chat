"""
Intent Patterns and Extractors

Regex definitions for intent recognition and parameter extraction.
"""

from ..models import IntentType

INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.DIRECTORY_LIST: [
        r"(show|list|see|view)\s*(the)?\s*(content|contents|folders|files)",
        r"(what('s|s| is)\s*(inside|in))\s*(the)?\s*(drive|folder)",
        r"(folders?|files?)\s*(inside|in)\s*(the)?\s*(drive|folder)",
        r"(list|show)\s*(folders?|files?)\s*(in|inside)\s*[a-z]:",
        r"(drive)\s*[a-z]\s*(content|contents|folders|files)",
        r"(content|contents|contend)\s*(in|inside|of)\s*(the)?\s*[a-z]\b",
    ],
    IntentType.SYSTEM_INFO: [
        r"^system\s*(info|information|status)$",
        r"system\s*(usage|utilization)",
        r"get\s*system\s*(info|information|status|usage)",
        r"show\s*(my\s*)?system\s*info",
        r"get_system_info",
        r"(cpu|processor)\s*(usage|load|status)?",
        r"(ram|memory)\s*(usage|status)?",
        r"(storage|disk)\s*(space|usage|status)",
        r"drive\s*(space|usage|status)",
        r"(network|internet)\s*(status|connection)$",
        r"(what|which)\s*(network|wifi|wi-fi|internet).*(connect|connected|connecting|status)",
        r"how\s*(much|many)\s*(memory|ram|storage|space)",
        r"what('s|s| is)\s*(the|my)?\s*(cpu|memory|ram|storage)",
    ],
    IntentType.VOLUME_CONTROL: [
        r"(volume|sound)\s*(up|down|louder|quieter|higher|lower)",
        r"(turn|set)\s*(up|down)?\s*(the)?\s*volume",
        r"(mute|unmute)(\s+(the|my|the\s+)?)?(\s+(sound|volume|audio|music))?",
        r"(increase|decrease|raise|lower)\s*(the)?\s*volume",
        r"adjust\s*(the)?\s*volume(\s*(to|at)\s*\d+)?",
        r"volume\s*(to|at)?\s*(\d+)",
        r"(set|change)\s*(the)?\s*volume\s*(to|at)?\s*(\d+)",
        r"make\s*it\s*(louder|quieter)",
        r"what\s*is\s*(my|the)\s*(current\s*)?(system\s*)?volume",
    ],
    IntentType.MUSIC_CONTROL: [
        r"(play|pause|stop)\s*(music|song|track|spotify)?",
        r"(next|previous|skip)\s*(track|song)?",
        r"(spotify|music)\s*(play|pause|next|previous|skip)",
        r"what('s|s| is)?\s*(music|song|track)?\s*(is\s*)?playing",
        r"current\s*(track|song)",
        r"resume\s*(music|playback|spotify)?",
    ],
    IntentType.SPOTIFY_INFO: [
        r"show\s*(my|the)?\s*(spotify\s*)?queue",
        r"list\s*(my)?\s*(spotify\s*)?playlists?",
        r"add\s+(.*)\s+to\s+(the\s+)?queue",
        r"search\s+(for\s+)?(.*)\s+on\s+spotify",
        r"find\s+(the\s+)?(.*)\s+on\s+spotify",
        r"what\s+song\s+am\s+i\s+listening\s+to",
        r"show\s+available\s+spotify\s+devices",
        r"show\s+tracks\s+in\s+(my\s+)?playlist",
        r"play\s+(my\s+)?playlist",
    ],
    IntentType.NETWORK_TOGGLE: [
        r"(turn|toggle|switch)\s*(on|off)\s*(the)?\s*(wifi|wi-fi|bluetooth|ethernet)",
        r"(enable|disable)\s*(the)?\s*(wifi|wi-fi|bluetooth|ethernet)",
        r"(wifi|wi-fi|bluetooth|ethernet)\s*(on|off)",
        r"(connect|disconnect)\s*(to)?\s*(wifi|wi-fi|bluetooth|ethernet)",
    ],
    IntentType.FILE_ORGANIZATION: [
        r"(organize|clean|sort|tidy)\s*(my|the)?\s*(downloads|desktop|documents|folder|files)",
        r"(organize|sort)\s*(files|folder)",
        r"(clean up|tidy)\s*(downloads|desktop|documents|folder)?",
        r"(arrange|group)\s*(files)\s*(by)?\s*(type|extension|date)",
    ],
    IntentType.PATH_RESOLVE: [
        r"where\s*(is|at)\s*(my|the)?\s*(downloads|documents|desktop|home|project)\s*(folder|directory)?",
        r"resolve\s*(the)?\s*path\s*(for|to)?",
    ],
    IntentType.BLUETOOTH_CONTROL: [
        r"(list|show|see|find)\s*(my)?\s*bluetooth\s*(devices?)?",
        r"(connect|disconnect)\s*(to|from)?\s*(the)?\s*(bluetooth\s*)?device",
    ],
}

PARAMETER_EXTRACTORS = {
    IntentType.VOLUME_CONTROL: {
        "direction": [
            (r"\bunmute\b", "unmute"),
            (r"\bmute\b", "mute"),
            (r"(up|louder|higher|increase|raise)", "up"),
            (r"(down|quieter|lower|decrease)", "down"),
        ],
        "level": r"(\d+)\s*(%|percent)?",
    },
    IntentType.MUSIC_CONTROL: {
        "action": [
            (r"\b(play|resume)\b", "play"),
            (r"\b(pause|stop)\b", "pause"),
            (r"\b(next|skip)\b", "next"),
            (r"\bprevious\b", "previous"),
            (r"(what('s|s| is)?\s*(music|song|track)?\s*(is\s*)?playing|current\s*(track|song))", "current"),
        ],
    },
    IntentType.NETWORK_TOGGLE: {
        "device": [
            (r"\b(wifi|wi-fi)\b", "wifi"),
            (r"\bbluetooth\b", "bluetooth"),
            (r"\bethernet\b", "ethernet"),
        ],
        "state": [
            (r"\b(on|enable|connect)\b", "on"),
            (r"\b(off|disable|disconnect)\b", "off"),
        ],
    },
    IntentType.FILE_ORGANIZATION: {
        "target_folder": [
            (r"\bdownloads?\b", "downloads"),
            (r"\bdesktop\b", "desktop"),
            (r"\bdocuments?\b", "documents"),
        ],
        "strategy": [
            (r"\b(date|month|year)\b", "date"),
            (r"\b(type|category)\b", "type"),
            (r"\b(extension|ext)\b", "extension"),
        ],
        "dry_run": [
            (r"\b(preview|dry\s*run|simulate)\b", "true"),
            (r"\b(now|apply|execute|do it)\b", "false"),
        ],
    },
    IntentType.DIRECTORY_LIST: {
        "include_hidden": [
            (r"\b(hidden|all files)\b", "true"),
        ],
    },
    IntentType.PATH_RESOLVE: {
        "name": [
            (r"\bdownloads?\b", "downloads"),
            (r"\bdocuments?\b", "documents"),
            (r"\bdesktop\b", "desktop"),
            (r"\bhome\b", "home"),
            (r"\bproject\b", "project"),
        ],
    },
}
