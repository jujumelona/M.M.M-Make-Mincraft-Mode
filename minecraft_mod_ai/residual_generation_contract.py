from __future__ import annotations

"""Residual Generation Contract & Write Protection Gate.

Defines the exact, bounded contract for residual code/asset generation. Coder agents
and generation tools are strictly gated by this contract: protected artifacts cannot
be overwritten or modified, and generation is confined to declared missing interfaces,
unbound registries, missing resources, and necessary glue bindings.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class ProtectedReuseArtifactError(PermissionError):
    """Raised when an edit attempts to modify or overwrite a protected reused artifact."""

    def __init__(self, path: str, message: str = "") -> None:
        super().__init__(message or f"Protected reuse artifact cannot be modified: {path}")
        self.path = path


class ResidualScopeViolation(PermissionError):
    """Raised when a generated file path is outside the allowed residual generation scope."""

    def __init__(self, path: str, message: str = "") -> None:
        super().__init__(message or f"Path is outside allowed residual generation scope: {path}")
        self.path = path


@dataclass(frozen=True)
class ResourceRequirement:
    logical_id: str
    resource_type: str  # "texture" | "model" | "sound" | "lang" | "data"
    target_path: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "resource_type": self.resource_type,
            "target_path": self.target_path,
            "description": self.description,
        }


@dataclass(frozen=True)
class RegistryRequirement:
    registry_key: str  # e.g., "minecraft:item", "minecraft:entity_type"
    entry_id: str      # e.g., "my_mod:boss_entity"
    backing_class: str # e.g., "ai.minecraft.generated.rpg.BossEntity"
    is_bound: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_key": self.registry_key,
            "entry_id": self.entry_id,
            "backing_class": self.backing_class,
            "is_bound": self.is_bound,
        }


@dataclass(frozen=True)
class EntrypointRequirement:
    environment: str  # "main" | "client" | "server"
    entrypoint_class: str
    interface_type: str = "ModInitializer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "entrypoint_class": self.entrypoint_class,
            "interface_type": self.interface_type,
        }


@dataclass(frozen=True)
class DependencyRequirement:
    coordinate: str
    configuration: str = "implementation"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "configuration": self.configuration,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GlueContract:
    target_symbol: str
    caller_symbol: str
    purpose: str
    suggested_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_symbol": self.target_symbol,
            "caller_symbol": self.caller_symbol,
            "purpose": self.purpose,
            "suggested_file": self.suggested_file,
        }


@dataclass(frozen=True)
class ResidualGenerationContract:
    capability: str
    requirement_ids: tuple[str, ...] = ()
    protected_artifacts: Mapping[str, str] = field(default_factory=dict)  # path -> sha256
    protected_symbols: tuple[str, ...] = ()
    allowed_write_paths: tuple[str, ...] = ()
    allowed_create_prefixes: tuple[str, ...] = ("src/main/java/", "src/main/resources/", "src/client/java/")
    required_new_artifacts: tuple[str, ...] = ()
    required_symbols: tuple[str, ...] = ()
    required_interfaces: tuple[str, ...] = ()
    required_resource_edges: tuple[ResourceRequirement, ...] = ()
    required_registry_bindings: tuple[RegistryRequirement, ...] = ()
    required_entrypoint_changes: tuple[EntrypointRequirement, ...] = ()
    required_dependency_changes: tuple[DependencyRequirement, ...] = ()
    glue_contracts: tuple[GlueContract, ...] = ()

    def allows_path(self, path: str) -> bool:
        """Check if path is permissible for residual generation."""
        norm = path.replace("\\", "/").strip("/")
        if norm in self.protected_artifacts:
            return False
        if norm in self.allowed_write_paths:
            return True
        for pfx in self.allowed_create_prefixes:
            if norm.startswith(pfx):
                return True
        return False

    def validate_write(self, path: str, old_sha256: str | None = None) -> None:
        """Validate that a write/edit operation complies with this contract."""
        validate_residual_write(path, old_sha256, self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/residual-generation-contract-v1",
            "capability": self.capability,
            "requirement_ids": list(self.requirement_ids),
            "protected_artifacts": dict(self.protected_artifacts),
            "protected_symbols": list(self.protected_symbols),
            "allowed_write_paths": list(self.allowed_write_paths),
            "allowed_create_prefixes": list(self.allowed_create_prefixes),
            "required_new_artifacts": list(self.required_new_artifacts),
            "required_symbols": list(self.required_symbols),
            "required_interfaces": list(self.required_interfaces),
            "required_resource_edges": [r.to_dict() for r in self.required_resource_edges],
            "required_registry_bindings": [r.to_dict() for r in self.required_registry_bindings],
            "required_entrypoint_changes": [e.to_dict() for e in self.required_entrypoint_changes],
            "required_dependency_changes": [d.to_dict() for d in self.required_dependency_changes],
            "glue_contracts": [g.to_dict() for g in self.glue_contracts],
        }


def validate_residual_write(
    path: str,
    old_sha256: str | None,
    contract: ResidualGenerationContract,
) -> None:
    """Validate write request against the ResidualGenerationContract."""
    norm = path.replace("\\", "/").strip("/")
    if norm in contract.protected_artifacts:
        raise ProtectedReuseArtifactError(norm)
    if not contract.allows_path(norm):
        raise ResidualScopeViolation(norm)
