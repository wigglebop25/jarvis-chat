"""
Tool result and error formatting.

Handles formatting of tool execution results and errors
for consistent presentation to LLM and user.
"""

from typing import Any


def format_tool_result(tool_name: str, result: Any) -> str:
    """Format a tool execution result into a human-readable string."""
    if tool_name == "get_system_info" and isinstance(result, dict):
        lines: list[str] = []

        if "cpu" in result:
            cpu = result.get("cpu")
            cpu_text = f"{float(cpu):.2f}" if isinstance(cpu, (int, float)) else str(cpu)
            lines.append(f"CPU: {cpu_text}%")

        ram = result.get("ram")
        if isinstance(ram, dict):
            ram_used = ram.get("used_gb", "N/A")
            ram_total = ram.get("total_gb", "N/A")
            ram_pct = ram.get("percent", "N/A")
            lines.append(f"RAM: {ram_used}GB/{ram_total}GB ({ram_pct}%)")

        storage = result.get("storage")
        if isinstance(storage, list) and storage:
            lines.append("Storage:")
            for part in storage[:12]:
                if not isinstance(part, dict):
                    continue
                mount = part.get("mount", "unknown")
                used = part.get("used_gb", "N/A")
                total = part.get("total_gb", "N/A")
                free = part.get("free_gb", "N/A")
                percent = part.get("percent", "N/A")
                lines.append(f"- {mount}: {used}GB/{total}GB used ({percent}%), {free}GB free")

        network = result.get("network")
        if isinstance(network, dict):
            interface = network.get("interface") or "unknown"
            connected = bool(network.get("connected"))
            state = "connected" if connected else "disconnected"
            lines.append(f"Network: {state} ({interface})")

        if lines:
            return "\n".join(lines)
        return "No system information was returned."

    if tool_name == "control_volume" and isinstance(result, dict):
        if "level" in result:
            return f"Volume set to {result['level']}%."
        if "muted" in result:
            return "Muted." if result["muted"] else "Unmuted."
        return "Volume command completed."

    if tool_name == "control_spotify" and isinstance(result, dict):
        # Handle check_auth responses with login URL (from Rust MCP server)
        if "login_url" in result and not result.get("authenticated"):
            return (
                "You are not currently logged into Spotify. "
                "Please open this link in your browser to authenticate:\n\n"
                f"{result['login_url']}\n\n"
                "After you log in and grant permission, your Spotify account will be connected."
            )
        
        if result.get("authenticated"):
            profile = result.get("user_profile", {})
            if isinstance(profile, dict):
                user_name = profile.get("display_name", profile.get("email", "Unknown"))
                return f"You are logged into Spotify as: {user_name}"
            return "You are logged into Spotify."
        
        # Handle other Spotify action responses
        action = result.get("action", "")
        if action:
            return f"Spotify action completed: {action}."
        
        # If we have an error or message field, return that
        if "error" in result:
            return f"Spotify error: {result['error']}"
        if "message" in result:
            return result['message']
        
        return "Spotify action completed."

    if tool_name == "toggle_network" and isinstance(result, dict):
        interface = result.get("interface", "network")
        enabled = result.get("enabled", False)
        return f"{interface} {'enabled' if enabled else 'disabled'}."

    if tool_name == "list_directory" and isinstance(result, dict):
        entries = result.get("entries", [])
        path_display = str(result.get("path", "the requested path"))
        if path_display.startswith("\\\\?\\"):
            path_display = path_display[4:]
        if not isinstance(entries, list) or not entries:
            return f"No visible entries found in {path_display}."
        preview = []
        for item in entries[:20]:
            if isinstance(item, dict):
                name = item.get("name", "")
                entry_type = item.get("type", "file")
                marker = "[DIR]" if entry_type == "directory" else "[FILE]"
                preview.append(f"{marker} {name}")
        suffix = " (truncated)" if result.get("truncated") else ""
        return (
            f"Directory entries in {path_display}{suffix}:\n"
            + "\n".join(preview)
        )

    return str(result)


def format_tool_error(tool_name: str, error: str) -> str:
    """Format a tool execution error into a user-friendly message."""
    if tool_name == "toggle_network":
        if "Missing required field: interface" in error:
            return (
                "I couldn't complete that action: please specify which interface "
                "(wifi, bluetooth, or ethernet)."
            )
        if "Missing required field: enable" in error:
            return "I couldn't complete that action: please specify on or off."
    return f"I couldn't complete that action: {error or 'unknown error'}"
