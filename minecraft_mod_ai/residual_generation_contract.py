from __future__ import annotations

"""Residual Generation Contract & Write Protection Gate.

Defines the exact, bounded contract for residual code/asset generation. Coder agents
and generation tools are strictly gated by this contract: protected artifacts cannot
be overwritten or modified, and generation is confined to declared missing interfaces,
unbound registries, missing resources, and necessary glue bindings.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_WORKSPACE_CONTRACT_PATH = ".minecraft_ai/residual-generation-contracts.json"


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


class ResidualWritePreconditionError(PermissionError):
    """Raised when a residual replacement is not bound to the planned file bytes."""

    def __init__(self, path: str, message: str = "") -> None:
        super().__init__(
            message or f"Residual write precondition does not match planned bytes: {path}"
        )
        self.path = path


class ResidualContractLoadError(ValueError):
    """Raised when persisted residual write policy is malformed."""


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
    expected_old_sha256: Mapping[str, str] = field(default_factory=dict)
    allowed_create_prefixes: tuple[str, ...] = ()
    required_new_artifacts: tuple[str, ...] = ()
    required_symbols: tuple[str, ...] = ()
    required_interfaces: tuple[str, ...] = ()
    required_resource_edges: tuple[ResourceRequirement, ...] = ()
    required_registry_bindings: tuple[RegistryRequirement, ...] = ()
    required_entrypoint_changes: tuple[EntrypointRequirement, ...] = ()
    required_dependency_changes: tuple[DependencyRequirement, ...] = ()
    glue_contracts: tuple[GlueContract, ...] = ()

    def __post_init__(self) -> None:
        if self.allowed_create_prefixes:
            raise ValueError(
                "Residual create permissions must name exact artifacts, not prefixes."
            )
        protected: dict[str, str] = {}
        for raw_path, raw_digest in self.protected_artifacts.items():
            path = _normalize_path(raw_path)
            digest = _normalize_sha256(raw_digest)
            if not path or not digest or path in protected:
                raise ValueError(
                    "Protected reuse artifacts require unique safe SHA-256 paths."
                )
            protected[path] = digest

        write_paths = tuple(_normalize_path(path) for path in self.allowed_write_paths)
        new_paths = tuple(_normalize_path(path) for path in self.required_new_artifacts)
        if (
            not all(write_paths)
            or not all(new_paths)
            or len(set(write_paths)) != len(write_paths)
            or len(set(new_paths)) != len(new_paths)
            or set(write_paths) & set(new_paths)
        ):
            raise ValueError(
                "Residual writes and creates require disjoint unique exact paths."
            )

        expected: dict[str, str] = {}
        for raw_path, raw_digest in self.expected_old_sha256.items():
            path = _normalize_path(raw_path)
            digest = _normalize_sha256(raw_digest)
            if not path or not digest or path in expected:
                raise ValueError(
                    "Residual replacement preconditions require unique SHA-256 paths."
                )
            expected[path] = digest
        if set(expected) != set(write_paths):
            raise ValueError(
                "Every residual replacement path must carry its exact old SHA-256."
            )
        if set(protected) & (set(write_paths) | set(new_paths)):
            raise ValueError("Protected reuse artifacts cannot be residual write targets.")

        object.__setattr__(self, "protected_artifacts", protected)
        object.__setattr__(self, "allowed_write_paths", tuple(sorted(write_paths)))
        object.__setattr__(self, "expected_old_sha256", dict(sorted(expected.items())))
        object.__setattr__(self, "required_new_artifacts", tuple(sorted(new_paths)))

    def allows_path(self, path: str) -> bool:
        """Check if path is permissible for residual generation."""
        norm = _normalize_path(path)
        if not norm:
            return False
        if norm in self.protected_artifacts:
            return False
        if norm in self.allowed_write_paths or norm in self.required_new_artifacts:
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
            "expected_old_sha256": dict(self.expected_old_sha256),
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
    norm = _normalize_path(path)
    if not norm:
        raise ResidualScopeViolation(str(path), "Residual write path is unsafe or empty.")
    if norm in contract.protected_artifacts:
        raise ProtectedReuseArtifactError(norm)
    if not contract.allows_path(norm):
        raise ResidualScopeViolation(norm)

    expected_by_path = {
        _normalize_path(item_path): _normalize_sha256(digest)
        for item_path, digest in contract.expected_old_sha256.items()
    }
    expected = expected_by_path.get(norm, "")
    if old_sha256 is None:
        if expected or norm in {
            _normalize_path(item) for item in contract.allowed_write_paths
        }:
            raise ResidualWritePreconditionError(
                norm,
                f"Residual path already exists and requires its planned old SHA-256: {norm}",
            )
        return

    actual = _normalize_sha256(old_sha256)
    if not expected:
        raise ResidualWritePreconditionError(
            norm,
            f"Residual replacement has no code-owned old SHA-256 precondition: {norm}",
        )
    if not actual or actual != expected:
        raise ResidualWritePreconditionError(norm)
    if norm not in {_normalize_path(item) for item in contract.allowed_write_paths}:
        raise ResidualScopeViolation(
            norm,
            f"Residual replacement is not an exact allowed write path: {norm}",
        )


def validate_residual_write_against_contracts(
    path: str,
    old_sha256: str | None,
    contracts: tuple[ResidualGenerationContract, ...] | list[ResidualGenerationContract],
) -> None:
    """Apply the single residual write gate across one plan's contracts.

    A protected path remains protected even if another contract accidentally lists
    it as writable.  Otherwise exactly one declared residual scope must authorize
    the operation; undeclared source edits fail closed.
    """

    normalized = _normalize_path(path)
    if not normalized:
        raise ResidualScopeViolation(str(path), "Residual write path is unsafe or empty.")
    for contract in contracts:
        if normalized in {
            _normalize_path(protected) for protected in contract.protected_artifacts
        }:
            validate_residual_write(normalized, old_sha256, contract)

    matching = [contract for contract in contracts if contract.allows_path(normalized)]
    if not matching:
        raise ResidualScopeViolation(
            normalized,
            "Residual write is not declared by the active residual contract.",
        )
    if len(matching) != 1:
        raise ResidualScopeViolation(
            normalized,
            "Residual write is claimed by multiple contracts and is ambiguous.",
        )
    validate_residual_write(normalized, old_sha256, matching[0])


def load_residual_generation_contracts(
    project_root: str | Path,
) -> tuple[ResidualGenerationContract, ...]:
    """Load the planner-owned residual policy persisted in the workspace manifest."""

    path = Path(project_root) / _WORKSPACE_CONTRACT_PATH
    if not path.exists():
        return ()
    if path.is_symlink() or not path.is_file():
        raise ResidualContractLoadError("Residual contract metadata is not a regular file.")
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidualContractLoadError("Residual contract metadata is unreadable.") from exc
    raw_contracts = payload.get("contracts") if isinstance(payload, Mapping) else None
    if not isinstance(raw_contracts, list):
        raise ResidualContractLoadError("Residual contract metadata has no contracts array.")

    contracts: list[ResidualGenerationContract] = []
    for raw in raw_contracts:
        if not isinstance(raw, Mapping):
            raise ResidualContractLoadError("Residual contract entry must be an object.")
        capability = str(raw.get("capability") or "").strip()
        if not capability:
            raise ResidualContractLoadError("Residual contract capability must be non-empty.")
        for key in (
            "protected_artifacts",
            "expected_old_sha256",
        ):
            if not isinstance(raw.get(key, {}), Mapping):
                raise ResidualContractLoadError(f"Residual contract field {key} must be an object.")
        sequence_fields = (
            "requirement_ids",
            "protected_symbols",
            "allowed_write_paths",
            "allowed_create_prefixes",
            "required_new_artifacts",
        )
        if any(
            not isinstance(raw.get(key, ()), list)
            for key in sequence_fields
        ):
            raise ResidualContractLoadError("Residual contract path fields must be arrays.")
        try:
            contracts.append(
                ResidualGenerationContract(
                    capability=capability,
                    requirement_ids=tuple(str(item) for item in raw.get("requirement_ids", ())),
                    protected_artifacts={
                        str(item_path): str(digest)
                        for item_path, digest in raw.get("protected_artifacts", {}).items()
                    },
                    protected_symbols=tuple(str(item) for item in raw.get("protected_symbols", ())),
                    allowed_write_paths=tuple(str(item) for item in raw.get("allowed_write_paths", ())),
                    expected_old_sha256={
                        str(item_path): str(digest)
                        for item_path, digest in raw.get("expected_old_sha256", {}).items()
                    },
                    allowed_create_prefixes=tuple(str(item) for item in raw.get("allowed_create_prefixes", ())),
                    required_new_artifacts=tuple(str(item) for item in raw.get("required_new_artifacts", ())),
                )
            )
        except ValueError as exc:
            raise ResidualContractLoadError(str(exc)) from exc
    return tuple(contracts)


def _normalize_path(path: Any) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        return ""
    parts = tuple(part for part in raw.split("/") if part not in {"", "."})
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def _normalize_sha256(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", raw):
        return "sha256:" + raw
    if re.fullmatch(r"sha256:[0-9a-f]{64}", raw):
        return raw
    return ""
