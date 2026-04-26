"""Convert tool schemas between different LLM provider formats."""

from typing import Any


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
