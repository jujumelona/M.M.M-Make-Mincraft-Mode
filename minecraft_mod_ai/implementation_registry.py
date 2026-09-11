"""Implementation registry - single source of truth for implementation hashes.

This module provides the ImplementationRegistry class that:
1. Registers templates, Python executors, validators, and schemas
2. Computes SHA-256 hashes from actual content (not fake string literals)
3. Provides implementation_id references for leaf bindings
4. Enables runtime verification by re-hashing before execution

This replaces all scattered hash calculation with a single authority.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .implementation_identity import (
    ExecutorType,
    ImplementationId,
    SchemaRegistration,
    ValidatorRegistration,
    ValidatorType,
    compute_content_hash,
    compute_json_schema_hash,
    compute_source_hash,
    current_timestamp,
)


class ImplementationRegistryError(Exception):
    """Base exception for implementation registry errors."""
    pass


class ImplementationRegistry:
    """Central registry for all implementations, validators, and schemas.
    
    This is the single source of truth for:
    - Implementation content hashes (templates, Python generators, etc.)
    - Validator source hashes
    - Type schema hashes
    
    All other components should reference implementations by ID,
    not calculate hashes independently.
    """
    
    def __init__(self):
        self._implementations: dict[str, ImplementationId] = {}
        self._validators: dict[str, ValidatorRegistration] = {}
        self._schemas: dict[str, SchemaRegistration] = {}
        
        # Reverse lookup: content_hash -> implementation_id
        self._hash_to_impl: dict[str, str] = {}
    
    def register_template(
        self,
        template_id: str,
        template_path: Path,
    ) -> ImplementationId:
        """Register a YAML template by reading and hashing its content.
        
        Args:
            template_id: Unique identifier (e.g., "fabric/item/register_basic")
            template_path: Path to template YAML file
            
        Returns:
            ImplementationId with content hash
            
        Raises:
            ImplementationRegistryError: If template_id already registered with different hash
        """
        if not template_path.exists():
            raise ImplementationRegistryError(
                f"Template file not found: {template_path}"
            )
        
        # Read actual template bytes and hash
        content_bytes = template_path.read_bytes()
        content_hash = compute_content_hash(content_bytes)
        
        return self._register_implementation(
            implementation_id=template_id,
            content_hash=content_hash,
            executor_type=ExecutorType.TEMPLATE,
            source_reference=str(template_path),
        )
    
    def register_python_executor(
        self,
        executor_id: str,
        executor_fn: Callable,
    ) -> ImplementationId:
        """Register a Python function executor by hashing its source.
        
        Args:
            executor_id: Unique identifier (e.g., "entity_generator")
            executor_fn: Python function that implements the executor
            
        Returns:
            ImplementationId with source hash
        """
        try:
            source_code = inspect.getsource(executor_fn)
        except (TypeError, OSError) as exc:
            raise ImplementationRegistryError(
                f"Cannot extract source for {executor_id}: {exc}"
            ) from exc
        
        source_hash = compute_source_hash(source_code)
        
        module = inspect.getmodule(executor_fn)
        module_name = module.__name__ if module else "unknown"
        function_name = executor_fn.__name__
        
        return self._register_implementation(
            implementation_id=executor_id,
            content_hash=source_hash,
            executor_type=ExecutorType.PYTHON_GENERATOR,
            source_reference=f"{module_name}.{function_name}",
        )
    
    def register_validator(
        self,
        validator_id: str,
        validator_fn: Callable,
        validator_type: ValidatorType,
    ) -> ValidatorRegistration:
        """Register a validator function by hashing its source.
        
        Args:
            validator_id: Unique identifier (e.g., "java_syntax")
            validator_fn: Validator function
            validator_type: Type of validation performed
            
        Returns:
            ValidatorRegistration with source hash
        """
        try:
            source_code = inspect.getsource(validator_fn)
        except (TypeError, OSError) as exc:
            raise ImplementationRegistryError(
                f"Cannot extract source for validator {validator_id}: {exc}"
            ) from exc
        
        source_hash = compute_source_hash(source_code)
        
        module = inspect.getmodule(validator_fn)
        module_name = module.__name__ if module else "unknown"
        function_name = validator_fn.__name__
        
        registration = ValidatorRegistration(
            validator_id=validator_id,
            validator_type=validator_type,
            source_hash=source_hash,
            function_reference=f"{module_name}.{function_name}",
            registered_at=current_timestamp(),
        )
        
        # Check for conflicts
        if validator_id in self._validators:
            existing = self._validators[validator_id]
            if existing.source_hash != source_hash:
                raise ImplementationRegistryError(
                    f"Validator {validator_id} already registered with different hash: "
                    f"{existing.source_hash[:8]} != {source_hash[:8]}"
                )
            return existing
        
        self._validators[validator_id] = registration
        return registration
    
    def register_json_schema(
        self,
        type_id: str,
        schema: dict[str, Any],
        definition_reference: str,
    ) -> SchemaRegistration:
        """Register a JSON Schema by computing its canonical hash.
        
        Args:
            type_id: Type name (e.g., "EntityRenderDesign")
            schema: JSON Schema dict
            definition_reference: Reference to schema definition location
            
        Returns:
            SchemaRegistration with schema hash
        """
        schema_hash = compute_json_schema_hash(schema)
        
        registration = SchemaRegistration(
            type_id=type_id,
            schema_hash=schema_hash,
            schema_type="json_schema",
            definition_reference=definition_reference,
            registered_at=current_timestamp(),
        )
        
        # Check for conflicts
        if type_id in self._schemas:
            existing = self._schemas[type_id]
            if existing.schema_hash != schema_hash:
                raise ImplementationRegistryError(
                    f"Schema {type_id} already registered with different hash: "
                    f"{existing.schema_hash[:8]} != {schema_hash[:8]}"
                )
            return existing
        
        self._schemas[type_id] = registration
        return registration
    
    def register_pydantic_schema(
        self,
        type_id: str,
        model_class: type,
    ) -> SchemaRegistration:
        """Register a Pydantic model by converting to JSON Schema and hashing.
        
        Args:
            type_id: Type name (matches model class name typically)
            model_class: Pydantic BaseModel subclass
            
        Returns:
            SchemaRegistration with schema hash
        """
        try:
            # Pydantic v2 API
            if hasattr(model_class, "model_json_schema"):
                schema = model_class.model_json_schema()
            # Pydantic v1 API
            elif hasattr(model_class, "schema"):
                schema = model_class.schema()
            else:
                raise ImplementationRegistryError(
                    f"{model_class} is not a Pydantic model"
                )
        except Exception as exc:
            raise ImplementationRegistryError(
                f"Cannot extract schema from {model_class}: {exc}"
            ) from exc
        
        schema_hash = compute_json_schema_hash(schema)
        
        module = inspect.getmodule(model_class)
        module_name = module.__name__ if module else "unknown"
        class_name = model_class.__name__
        
        registration = SchemaRegistration(
            type_id=type_id,
            schema_hash=schema_hash,
            schema_type="pydantic",
            definition_reference=f"{module_name}.{class_name}",
            registered_at=current_timestamp(),
        )
        
        # Check for conflicts
        if type_id in self._schemas:
            existing = self._schemas[type_id]
            if existing.schema_hash != schema_hash:
                raise ImplementationRegistryError(
                    f"Schema {type_id} already registered with different hash: "
                    f"{existing.schema_hash[:8]} != {schema_hash[:8]}"
                )
            return existing
        
        self._schemas[type_id] = registration
        return registration
    
    def get_implementation(self, implementation_id: str) -> ImplementationId:
        """Get registered implementation by ID.
        
        Raises:
            ImplementationRegistryError: If not found
        """
        if implementation_id not in self._implementations:
            raise ImplementationRegistryError(
                f"Implementation not registered: {implementation_id}"
            )
        return self._implementations[implementation_id]
    
    def get_validator(self, validator_id: str) -> ValidatorRegistration:
        """Get registered validator by ID.
        
        Raises:
            ImplementationRegistryError: If not found
        """
        if validator_id not in self._validators:
            raise ImplementationRegistryError(
                f"Validator not registered: {validator_id}"
            )
        return self._validators[validator_id]
    
    def get_schema(self, type_id: str) -> SchemaRegistration:
        """Get registered schema by type ID.
        
        Raises:
            ImplementationRegistryError: If not found
        """
        if type_id not in self._schemas:
            raise ImplementationRegistryError(
                f"Schema not registered: {type_id}"
            )
        return self._schemas[type_id]
    
    def verify_implementation_hash(
        self,
        implementation_id: str,
        expected_hash: str,
    ) -> bool:
        """Verify that current implementation hash matches expected.
        
        This is used at runtime to ensure implementation hasn't changed.
        """
        impl = self.get_implementation(implementation_id)
        return impl.content_sha256 == expected_hash
    
    def verify_validator_hash(
        self,
        validator_id: str,
        expected_hash: str,
    ) -> bool:
        """Verify that current validator hash matches expected."""
        validator = self.get_validator(validator_id)
        return validator.source_hash == expected_hash
    
    def verify_schema_hash(
        self,
        type_id: str,
        expected_hash: str,
    ) -> bool:
        """Verify that current schema hash matches expected."""
        schema = self.get_schema(type_id)
        return schema.schema_hash == expected_hash
    
    def list_implementations(self) -> list[ImplementationId]:
        """List all registered implementations."""
        return list(self._implementations.values())
    
    def list_validators(self) -> list[ValidatorRegistration]:
        """List all registered validators."""
        return list(self._validators.values())
    
    def list_schemas(self) -> list[SchemaRegistration]:
        """List all registered schemas."""
        return list(self._schemas.values())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize registry to dict for storage in catalog."""
        return {
            "implementations": {
                impl_id: impl.to_dict()
                for impl_id, impl in self._implementations.items()
            },
            "validators": {
                val_id: val.to_dict()
                for val_id, val in self._validators.items()
            },
            "schemas": {
                type_id: schema.to_dict()
                for type_id, schema in self._schemas.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationRegistry:
        """Deserialize registry from dict."""
        registry = cls()
        
        for impl_id, impl_data in data.get("implementations", {}).items():
            impl = ImplementationId.from_dict(impl_data)
            registry._implementations[impl_id] = impl
            registry._hash_to_impl[impl.content_sha256] = impl_id
        
        for val_id, val_data in data.get("validators", {}).items():
            val = ValidatorRegistration.from_dict(val_data)
            registry._validators[val_id] = val
        
        for type_id, schema_data in data.get("schemas", {}).items():
            schema = SchemaRegistration.from_dict(schema_data)
            registry._schemas[type_id] = schema
        
        return registry
    
    def _register_implementation(
        self,
        implementation_id: str,
        content_hash: str,
        executor_type: ExecutorType,
        source_reference: str,
    ) -> ImplementationId:
        """Internal: register an implementation.
        
        Checks for conflicts and maintains hash -> id reverse lookup.
        """
        impl = ImplementationId(
            implementation_id=implementation_id,
            content_sha256=content_hash,
            executor_type=executor_type,
            source_reference=source_reference,
            registered_at=current_timestamp(),
        )
        
        # Check for conflicts
        if implementation_id in self._implementations:
            existing = self._implementations[implementation_id]
            if existing.content_sha256 != content_hash:
                raise ImplementationRegistryError(
                    f"Implementation {implementation_id} already registered with different hash: "
                    f"{existing.content_sha256[:8]} != {content_hash[:8]}"
                )
            return existing
        
        self._implementations[implementation_id] = impl
        self._hash_to_impl[content_hash] = implementation_id
        return impl


# Global singleton registry
_global_registry: ImplementationRegistry | None = None


def get_global_registry() -> ImplementationRegistry:
    """Get the global implementation registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ImplementationRegistry()
    return _global_registry


def reset_global_registry():
    """Reset global registry (for testing)."""
    global _global_registry
    _global_registry = None
