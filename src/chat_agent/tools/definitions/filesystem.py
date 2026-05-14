"""Tool definitions - File system tools."""

FILE_TOOLS = [
    {
        "name": "list_directory",
        "description": "List files and folders in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include_hidden": {"type": "boolean"},
                "max_entries": {"type": "integer"},
                "directories_only": {"type": "boolean"},
                "files_only": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "organize_folder",
        "description": "Organize files by extension, type, or date.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "strategy": {"type": "string", "enum": ["extension", "type", "date"]},
                "recursive": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
                "include_hidden": {"type": "boolean"},
            },
            "required": ["path"],
        },
    },
]
