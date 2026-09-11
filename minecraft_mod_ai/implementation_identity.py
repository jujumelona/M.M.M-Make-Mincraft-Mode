"""Implementation identity and hash computation for HOST integrity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutorType(Enum):
    """Type of executor that runs an implementation."""
    TEMPLATE = "template"              # YAML template rendering (Jinja2-like)
    PYTHON_GENERATOR = "python_generator"  # Python function execution
    MODEL = "model"                    # LLM generation
    DETERMINISTIC = "deterministic"    # Pure deterministic function


class ValidatorType(Enum):
    """Type of validator for implementation verification."""
    JAVA_SYNTAX = "java_syntax"
    JSON_SCHEMA = "json_schema"
    COMPILE = "compile"
    GAMETEST = "gametest"
    MOD_INTEGRATION = "mod_integration_test"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ImplementationId:
    """Immutable identifier for an implementation with content hash.
    
    This is the single source of truth for implementation identity.
    All references to implementations should use this ID.
    """
    implementation_id: str      # Unique identifier (e.g., "fabric/item/register_basic")
    content_sha256: str         # SHA-256 hash of actual implementation bytes/source
    executor_type: ExecutorType
    source_reference: str       # Path to source file or function module.name
    registered_at: str          # ISO 8601 timestamp
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "content_sha256": self.content_sha256,
            "executor_type": self.executor_type.value,
            "source_reference": self.source_reference,
            "registered_at": self.registered_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationId:
        return cls(
            implementation_id=data["implementation_id"],
            content_sha256=data["content_sha256"],
            executor_type=ExecutorType(data["executor_type"]),
            source_reference=data["source_reference"],
            registered_at=data["registered_at"],
        )


@dataclass(frozen=True)
class ValidatorRegistration:
    """Registration info for a validator with source hash."""
    validator_id: str           # Unique identifier (e.g., "java_syntax")
    validator_type: ValidatorType
    source_hash: str            # SHA-256 of validator source code
    function_reference: str     # module.function for validator
    registered_at: str          # ISO 8601 timestamp
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "validator_type": self.validator_type.value,
            "source_hash": self.source_hash,
            "function_reference": self.function_reference,
            "registered_at": self.registered_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidatorRegistration:
        return cls(
            validator_id=data["validator_id"],
            validator_type=ValidatorType(data["validator_type"]),
            source_hash=data["source_hash"],
            function_reference=data["function_reference"],
            registered_at=data["registered_at"],
        )


@dataclass(frozen=True)
class SchemaRegistration:
    """Registration info for a type schema with canonical hash."""
    type_id: str                # Type name (e.g., "EntityRenderDesign")
    schema_hash: str            # SHA-256 of canonical JSON schema
    schema_type: str            # "json_schema" | "pydantic" | "dataclass"
    definition_reference: str   # module.ClassName or path to schema file
    registered_at: str          # ISO 8601 timestamp
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "schema_hash": self.schema_hash,
            "schema_type": self.schema_type,
            "definition_reference": self.definition_reference,
            "registered_at": self.registered_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaRegistration:
        return cls(
            type_id=data["type_id"],
            schema_hash=data["schema_hash"],
            schema_type=data["schema_type"],
            definition_reference=data["definition_reference"],
            registered_at=data["registered_at"],
        )


@dataclass(frozen=True)
class LeafImplementation:
    """Complete implementation binding for a canonical leaf.
    
    This ties together:
    - The implementation itself (template/generator/model)
    - All validators used to verify it
    - Input/output type schemas
    
    All references are by hash to ensure immutability and verifiability.
    """
    leaf_id: str                        # Canonical leaf identifier
    implementation: ImplementationId
    validator_hashes: dict[str, str]    # validator_id -> source_hash
    input_schema_hashes: dict[str, str] # port_name -> schema_hash
    output_schema_hashes: dict[str, str] # port_name -> schema_hash
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_id": self.leaf_id,
            "implementation": self.implementation.to_dict(),
            "validator_hashes": self.validator_hashes,
            "input_schema_hashes": self.input_schema_hashes,
            "output_schema_hashes": self.output_schema_hashes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeafImplementation:
        return cls(
            leaf_id=data["leaf_id"],
            implementation=ImplementationId.from_dict(data["implementation"]),
            validator_hashes=data["validator_hashes"],
            input_schema_hashes=data["input_schema_hashes"],
            output_schema_hashes=data["output_schema_hashes"],
        )


def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content bytes.
    
    This is the canonical way to hash implementation content.
    Always use this function to ensure consistency.
    """
    return hashlib.sha256(content).hexdigest()


def compute_source_hash(source_code: str) -> str:
    """Compute SHA-256 hash of Python source code.
    
    Normalizes line endings before hashing for consistency.
    """
    normalized = source_code.replace("\r\n", "\n").replace("\r", "\n")
    return compute_content_hash(normalized.encode("utf-8"))


def compute_json_schema_hash(schema: dict[str, Any]) -> str:
    """Compute SHA-256 hash of JSON schema.
    
    Uses canonical JSON encoding (sorted keys, no whitespace).
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return compute_content_hash(canonical.encode("utf-8"))


def current_timestamp() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return datetime.utcnow().isoformat() + "Z"
