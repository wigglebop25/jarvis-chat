"""Tool parameter validation."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ToolValidationError(ValueError):
    """Raised when tool parameter validation fails."""
    pass


class ToolParameterValidator:
    """Validates tool parameters against schema."""
    
    @staticmethod
    def validate_parameters(
        tool_definition: dict[str, Any],
        parameters: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against tool schema.
        
        Args:
            tool_definition: Tool definition with schema
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get required parameters
            properties = tool_definition.get("input_schema", {}).get("properties", {})
            required = tool_definition.get("input_schema", {}).get("required", [])
            
            # Check required parameters
            for req_param in required:
                if req_param not in parameters:
                    return False, f"Missing required parameter: {req_param}"
            
            # Check for unknown parameters
            for param_name in parameters.keys():
                if param_name not in properties:
                    logger.warning(f"Unknown parameter: {param_name}")
            
            # Validate parameter types (basic validation)
            for param_name, param_value in parameters.items():
                if param_name not in properties:
                    continue
                
                prop_schema = properties[param_name]
                expected_type = prop_schema.get("type", "string")
                
                if not _validate_type(param_value, expected_type):
                    return False, f"Invalid type for {param_name}: expected {expected_type}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, str(e)


def _validate_type(value: Any, expected_type: str) -> bool:
    """Validate value against expected type."""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    
    python_type = type_map.get(expected_type)
    if python_type is None:
        return True
    
    return isinstance(value, python_type)
