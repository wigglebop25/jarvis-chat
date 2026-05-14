"""Convert tool schemas between different LLM provider formats."""

from typing import Any, Tuple, Optional


class ToolSchemaConverter:
    """Convert provider-agnostic tool schemas to provider-specific formats."""

    @staticmethod
    def to_openai(tool: dict[str, Any]) -> dict[str, Any]:
        """
        Convert tool schema to OpenAI function format.

        Args:
            tool: Provider-agnostic tool schema

        Returns:
            OpenAI function schema
        """
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": ToolSchemaConverter._convert_parameters(tool.get("parameters", {})),
            },
        }

    @staticmethod
    def to_anthropic(tool: dict[str, Any]) -> dict[str, Any]:
        """
        Convert tool schema to Anthropic format.

        Args:
            tool: Provider-agnostic tool schema

        Returns:
            Anthropic tool schema
        """
        return {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "input_schema": ToolSchemaConverter._convert_parameters(tool.get("parameters", {})),
        }

    @staticmethod
    def to_gemini(tool: dict[str, Any]) -> dict[str, Any]:
        """
        Convert tool schema to Gemini format.

        Args:
            tool: Provider-agnostic tool schema

        Returns:
            Gemini tool schema (function declaration)
        """
        return {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": ToolSchemaConverter._convert_parameters(tool.get("parameters", {})),
        }

    @staticmethod
    def _convert_parameters(params: dict[str, Any]) -> dict[str, Any]:
        """
        Convert parameter schema to standard JSON Schema format.

        Args:
            params: Parameter schema with type and properties

        Returns:
            Converted parameter schema
        """
        schema = {
            "type": params.get("type", "object"),
        }

        if "properties" in params:
            schema["properties"] = {}
            for prop_name, prop_def in params["properties"].items():
                schema["properties"][prop_name] = ToolSchemaConverter._convert_property(prop_def)

        if "required" in params:
            schema["required"] = params["required"]

        return schema

    @staticmethod
    def _convert_property(prop_def: dict[str, Any]) -> dict[str, Any]:
        """
        Convert individual property definition.

        Args:
            prop_def: Property definition

        Returns:
            Converted property definition
        """
        converted = {
            "type": prop_def.get("type", "string"),
        }

        if "description" in prop_def:
            converted["description"] = prop_def["description"]

        if "enum" in prop_def:
            converted["enum"] = prop_def["enum"]

        if "default" in prop_def:
            converted["default"] = prop_def["default"]

        if "items" in prop_def:
            converted["items"] = ToolSchemaConverter._convert_property(prop_def["items"])

        return converted


# Runtime validation helpers
def to_jsonschema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return the tool's JSON Schema if available, otherwise a permissive schema."""
    # Prefer explicit 'parameters' or 'inputSchema' keys
    schema = tool.get("parameters") or tool.get("inputSchema")
    if isinstance(schema, dict):
        return schema
    # Fallback
    return {"type": "object", "additionalProperties": True}


def validate_tool_params(tool: dict[str, Any], params: dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate params against the tool's JSON Schema.

    Uses the 'jsonschema' package when available. If not present, performs a
    lightweight required-field check. Returns (is_valid, error_message).
    """
    schema = to_jsonschema(tool)

    try:
        import jsonschema  # type: ignore
        # Use dynamic access to avoid static analyzer errors on missing optional deps
        ValidationError = getattr(getattr(jsonschema, "exceptions", {}), "ValidationError", Exception)

        try:
            jsonschema.validate(instance=params or {}, schema=schema)
            return True, None
        except ValidationError as ve:
            return False, str(ve)
    except Exception:
        # Minimal fallback: check required keys
        required = schema.get("required") if isinstance(schema, dict) else None
        if not required:
            return True, None
        missing = [k for k in required if k not in (params or {})]
        if missing:
            return False, f"Missing required parameters: {missing}"
        return True, None


def assert_valid_tool_call(tool: dict[str, Any], params: dict[str, Any]) -> None:
    """Raise ValueError if params are invalid for the given tool."""
    ok, err = validate_tool_params(tool, params)
    if not ok:
        raise ValueError(f"Tool params validation failed for {tool.get('name')}: {err}")
