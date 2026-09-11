"""Type registry for runtime schema validation.

P0-8: Connect leaf type names to actual runtime validators.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .implementation_identity import compute_json_schema_hash


class TypeValidationError(Exception):
    """Error during type validation."""
    pass


class TypeRegistry:
    """Central registry for type schemas and validators.
    
    P0-8: This replaces string type names with actual runtime validation.
    """
    
    def __init__(self):
        self._validators: dict[str, tuple[Callable, str]] = {}  # type_id -> (validator, schema_hash)
        self._schemas: dict[str, dict[str, Any]] = {}  # type_id -> schema
    
    def register_json_schema(self, type_id: str, schema: dict[str, Any]):
        """Register a JSON Schema for a type.
        
        Args:
            type_id: Type identifier (e.g., "EntityRenderDesign")
            schema: JSON Schema dict
        """
        schema_hash = compute_json_schema_hash(schema)
        
        # Create validator from schema
        validator = self._create_json_schema_validator(schema)
        
        self._validators[type_id] = (validator, schema_hash)
        self._schemas[type_id] = schema
    
    def register_pydantic_model(self, type_id: str, model_class: type):
        """Register a Pydantic model for a type.
        
        Args:
            type_id: Type identifier
            model_class: Pydantic BaseModel subclass
        """
        # Extract schema from Pydantic model
        if hasattr(model_class, "model_json_schema"):
            schema = model_class.model_json_schema()
        elif hasattr(model_class, "schema"):
            schema = model_class.schema()
        else:
            raise TypeValidationError(f"{model_class} is not a Pydantic model")
        
        schema_hash = compute_json_schema_hash(schema)
        
        # Create validator that uses Pydantic's validation
        def pydantic_validator(value: Any) -> Any:
            if isinstance(value, model_class):
                return value
            # Parse and validate
            if hasattr(model_class, "model_validate"):
                return model_class.model_validate(value)
            else:
                return model_class.parse_obj(value)
        
        self._validators[type_id] = (pydantic_validator, schema_hash)
        self._schemas[type_id] = schema
    
    def register_custom_validator(
        self,
        type_id: str,
        validator: Callable[[Any], Any],
        schema: dict[str, Any],
    ):
        """Register a custom validator function.
        
        Args:
            type_id: Type identifier
            validator: Function that validates and returns typed value
            schema: JSON Schema describing the type
        """
        schema_hash = compute_json_schema_hash(schema)
        self._validators[type_id] = (validator, schema_hash)
        self._schemas[type_id] = schema
    
    def validate_input(self, type_id: str, value: Any) -> Any:
        """Validate input value against registered type.
        
        P0-8: This runs before executor invocation.
        
        Returns:
            Validated and possibly transformed value
            
        Raises:
            TypeValidationError: If validation fails
        """
        if type_id not in self._validators:
            raise TypeValidationError(f"Type not registered: {type_id}")
        
        validator, _ = self._validators[type_id]
        
        try:
            return validator(value)
        except Exception as exc:
            raise TypeValidationError(
                f"Input validation failed for type {type_id}: {exc}"
            ) from exc
    
    def validate_output(self, type_id: str, value: Any) -> Any:
        """Validate output value against registered type.
        
        P0-8: This runs after executor returns.
        
        Returns:
            Validated value
            
        Raises:
            TypeValidationError: If validation fails
        """
        if type_id not in self._validators:
            raise TypeValidationError(f"Type not registered: {type_id}")
        
        validator, _ = self._validators[type_id]
        
        try:
            return validator(value)
        except Exception as exc:
            raise TypeValidationError(
                f"Output validation failed for type {type_id}: {exc}"
            ) from exc
    
    def get_schema_hash(self, type_id: str) -> str:
        """Get schema hash for a type.
        
        P0-8: Used in implementation admission metadata.
        """
        if type_id not in self._validators:
            raise TypeValidationError(f"Type not registered: {type_id}")
        
        _, schema_hash = self._validators[type_id]
        return schema_hash
    
    def get_schema(self, type_id: str) -> dict[str, Any]:
        """Get JSON Schema for a type."""
        if type_id not in self._schemas:
            raise TypeValidationError(f"Type not registered: {type_id}")
        return self._schemas[type_id]
    
    def list_types(self) -> list[str]:
        """List all registered type IDs."""
        return list(self._validators.keys())
    
    def _create_json_schema_validator(self, schema: dict[str, Any]) -> Callable:
        """Create a validator function from JSON Schema.
        
        Uses jsonschema library if available, otherwise basic validation.
        """
        try:
            import jsonschema
            
            def validator(value: Any) -> Any:
                jsonschema.validate(instance=value, schema=schema)
                return value
            
            return validator
        except ImportError:
            # Fallback: basic type checking only
            def basic_validator(value: Any) -> Any:
                if "type" in schema:
                    expected_type = schema["type"]
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }
                    if expected_type in type_map:
                        expected = type_map[expected_type]
                        if not isinstance(value, expected):
                            raise TypeValidationError(
                                f"Expected {expected_type}, got {type(value).__name__}"
                            )
                return value
            
            return basic_validator


# Global type registry
_global_type_registry: TypeRegistry | None = None


def get_global_type_registry() -> TypeRegistry:
    """Get global type registry singleton."""
    global _global_type_registry
    if _global_type_registry is None:
        _global_type_registry = TypeRegistry()
    return _global_type_registry


def reset_global_type_registry():
    """Reset global type registry (for testing)."""
    global _global_type_registry
    _global_type_registry = None
