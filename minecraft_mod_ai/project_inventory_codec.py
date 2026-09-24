from __future__ import annotations

"""Strict codec for serialized project inventories.

Scanning and evidence construction remain in project_inventory. This module owns
untrusted mapping validation and reconstruction so parsing policy has one owner.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .project_inventory import (
    ComponentCatalog,
    ComponentRecord,
    DependencyRecord,
    EntryPointRecord,
    EvidenceLocator,
    ModMetadata,
    ProjectInventory,
    ProjectInventoryError,
    ProjectModule,
    ProjectTarget,
    SourceRoot,
    _component_id,
    _dependency_id,
    _evidence_id,
)


def _strict_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectInventoryError(f"{label} must be an object.")
    actual = {str(key) for key in value}
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ProjectInventoryError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ProjectInventoryError(f"{label} has unknown fields: {sorted(unknown)}")
    return value


def _strict_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ProjectInventoryError(f"{label} must be an array.")
    return value


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ProjectInventoryError(f"{label} must be a string.")
    return value


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProjectInventoryError(f"{label} must be an integer.")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ProjectInventoryError(f"{label} must be a boolean.")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    return tuple(
        _strict_string(item, f"{label}[]")
        for item in _strict_sequence(value, label)
    )


def _evidence_from_mapping(value: Any, label: str) -> EvidenceLocator:
    payload = _strict_mapping(
        value,
        {"locator_id", "path", "sha256", "size_bytes", "line_start", "line_end"},
        label,
    )
    result = EvidenceLocator(
        locator_id=_strict_string(payload["locator_id"], label + ".locator_id"),
        path=_strict_string(payload["path"], label + ".path"),
        sha256=_strict_string(payload["sha256"], label + ".sha256"),
        size_bytes=_strict_int(payload["size_bytes"], label + ".size_bytes"),
        line_start=_strict_int(payload["line_start"], label + ".line_start"),
        line_end=_strict_int(payload["line_end"], label + ".line_end"),
    )
    result.validate()
    expected_id = _evidence_id(
        result.path,
        result.sha256,
        result.line_start,
        result.line_end,
    )
    if result.locator_id != expected_id:
        raise ProjectInventoryError(
            f"{label} locator ID does not match its evidence payload."
        )
    return result


def _source_root_from_mapping(value: Any, label: str) -> SourceRoot:
    payload = _strict_mapping(
        value,
        {"module_id", "source_set", "language", "path", "generated", "test"},
        label,
    )
    return SourceRoot(
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        source_set=_strict_string(payload["source_set"], label + ".source_set"),
        language=_strict_string(payload["language"], label + ".language"),
        path=_strict_string(payload["path"], label + ".path"),
        generated=_strict_bool(payload["generated"], label + ".generated"),
        test=_strict_bool(payload["test"], label + ".test"),
    )


def _module_from_mapping(value: Any, label: str) -> ProjectModule:
    payload = _strict_mapping(
        value,
        {
            "module_id",
            "path",
            "build_files",
            "source_sets",
            "source_roots",
            "generated_resource_roots",
            "test_roots",
            "dependency_ids",
            "depends_on_modules",
        },
        label,
    )
    roots = tuple(
        _source_root_from_mapping(item, f"{label}.source_roots[{index}]")
        for index, item in enumerate(
            _strict_sequence(payload["source_roots"], label + ".source_roots")
        )
    )
    return ProjectModule(
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        path=_strict_string(payload["path"], label + ".path"),
        build_files=_string_tuple(payload["build_files"], label + ".build_files"),
        source_sets=_string_tuple(payload["source_sets"], label + ".source_sets"),
        source_roots=roots,
        generated_resource_roots=_string_tuple(
            payload["generated_resource_roots"],
            label + ".generated_resource_roots",
        ),
        test_roots=_string_tuple(payload["test_roots"], label + ".test_roots"),
        dependency_ids=_string_tuple(
            payload["dependency_ids"],
            label + ".dependency_ids",
        ),
        depends_on_modules=_string_tuple(
            payload["depends_on_modules"],
            label + ".depends_on_modules",
        ),
    )


def _dependency_from_mapping(value: Any, label: str) -> DependencyRecord:
    payload = _strict_mapping(
        value,
        {
            "dependency_id",
            "module_id",
            "configuration",
            "coordinate",
            "raw_coordinate",
            "group",
            "name",
            "version",
            "project_path",
            "resolved",
            "evidence",
        },
        label,
    )
    result = DependencyRecord(
        dependency_id=_strict_string(
            payload["dependency_id"],
            label + ".dependency_id",
        ),
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        configuration=_strict_string(
            payload["configuration"],
            label + ".configuration",
        ),
        coordinate=_strict_string(payload["coordinate"], label + ".coordinate"),
        raw_coordinate=_strict_string(
            payload["raw_coordinate"],
            label + ".raw_coordinate",
        ),
        group=_strict_string(payload["group"], label + ".group"),
        name=_strict_string(payload["name"], label + ".name"),
        version=_strict_string(payload["version"], label + ".version"),
        project_path=_strict_string(
            payload["project_path"],
            label + ".project_path",
        ),
        resolved=_strict_bool(payload["resolved"], label + ".resolved"),
        evidence=_evidence_from_mapping(payload["evidence"], label + ".evidence"),
    )
    expected_id = _dependency_id(
        module_id=result.module_id,
        configuration=result.configuration,
        coordinate=result.coordinate,
        project_path=result.project_path,
    )
    if result.dependency_id != expected_id:
        raise ProjectInventoryError(
            f"{label} dependency ID does not match its payload."
        )
    return result


def _entrypoint_from_mapping(value: Any, label: str) -> EntryPointRecord:
    payload = _strict_mapping(
        value,
        {"loader", "group", "value", "adapter", "evidence"},
        label,
    )
    return EntryPointRecord(
        loader=_strict_string(payload["loader"], label + ".loader"),
        group=_strict_string(payload["group"], label + ".group"),
        value=_strict_string(payload["value"], label + ".value"),
        adapter=_strict_string(payload["adapter"], label + ".adapter"),
        evidence=_evidence_from_mapping(payload["evidence"], label + ".evidence"),
    )


def _metadata_from_mapping(value: Any, label: str) -> ModMetadata:
    payload = _strict_mapping(
        value,
        {
            "loader",
            "path",
            "mod_id",
            "mod_name",
            "mod_version",
            "environment",
            "license",
            "entrypoints",
            "dependency_ids",
            "evidence",
        },
        label,
    )
    entrypoints = tuple(
        _entrypoint_from_mapping(item, f"{label}.entrypoints[{index}]")
        for index, item in enumerate(
            _strict_sequence(payload["entrypoints"], label + ".entrypoints")
        )
    )
    return ModMetadata(
        loader=_strict_string(payload["loader"], label + ".loader"),
        path=_strict_string(payload["path"], label + ".path"),
        mod_id=_strict_string(payload["mod_id"], label + ".mod_id"),
        mod_name=_strict_string(payload["mod_name"], label + ".mod_name"),
        mod_version=_strict_string(payload["mod_version"], label + ".mod_version"),
        environment=_strict_string(payload["environment"], label + ".environment"),
        license=_strict_string(payload["license"], label + ".license"),
        entrypoints=entrypoints,
        dependency_ids=_string_tuple(
            payload["dependency_ids"],
            label + ".dependency_ids",
        ),
        evidence=_evidence_from_mapping(payload["evidence"], label + ".evidence"),
    )


def _target_from_mapping(value: Any, label: str) -> ProjectTarget:
    payload = _strict_mapping(
        value,
        {
            "minecraft_versions",
            "loaders",
            "loader_versions",
            "java_versions",
            "mappings",
            "gradle_versions",
            "evidence",
        },
        label,
    )
    return ProjectTarget(
        minecraft_versions=_string_tuple(
            payload["minecraft_versions"],
            label + ".minecraft_versions",
        ),
        loaders=_string_tuple(payload["loaders"], label + ".loaders"),
        loader_versions=_string_tuple(
            payload["loader_versions"],
            label + ".loader_versions",
        ),
        java_versions=_string_tuple(
            payload["java_versions"],
            label + ".java_versions",
        ),
        mappings=_string_tuple(payload["mappings"], label + ".mappings"),
        gradle_versions=_string_tuple(
            payload["gradle_versions"],
            label + ".gradle_versions",
        ),
        evidence=tuple(
            _evidence_from_mapping(item, f"{label}.evidence[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["evidence"], label + ".evidence")
            )
        ),
    )


def _component_from_mapping(value: Any, label: str) -> ComponentRecord:
    payload = _strict_mapping(
        value,
        {
            "component_id",
            "kind",
            "name",
            "locator",
            "content_sha256",
            "module_id",
            "source_set",
            "side",
            "minecraft_versions",
            "loaders",
            "provides",
            "requires",
            "evidence",
            "provenance",
            "license_refs",
        },
        label,
    )
    result = ComponentRecord(
        component_id=_strict_string(
            payload["component_id"],
            label + ".component_id",
        ),
        kind=_strict_string(payload["kind"], label + ".kind"),
        name=_strict_string(payload["name"], label + ".name"),
        locator=_strict_string(payload["locator"], label + ".locator"),
        content_sha256=_strict_string(
            payload["content_sha256"],
            label + ".content_sha256",
        ),
        module_id=_strict_string(payload["module_id"], label + ".module_id"),
        source_set=_strict_string(payload["source_set"], label + ".source_set"),
        side=_strict_string(payload["side"], label + ".side"),
        minecraft_versions=_string_tuple(
            payload["minecraft_versions"],
            label + ".minecraft_versions",
        ),
        loaders=_string_tuple(payload["loaders"], label + ".loaders"),
        provides=_string_tuple(payload["provides"], label + ".provides"),
        requires=_string_tuple(payload["requires"], label + ".requires"),
        evidence=tuple(
            _evidence_from_mapping(item, f"{label}.evidence[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["evidence"], label + ".evidence")
            )
        ),
        provenance=_strict_string(payload["provenance"], label + ".provenance"),
        license_refs=_string_tuple(
            payload["license_refs"],
            label + ".license_refs",
        ),
    )
    expected_id = _component_id(
        kind=result.kind,
        name=result.name,
        locator=result.locator,
        module_id=result.module_id,
    )
    if result.component_id != expected_id:
        raise ProjectInventoryError(
            f"{label} component ID does not match its payload."
        )
    result.validate()
    return result


def _catalog_from_mapping(value: Any, label: str) -> ComponentCatalog:
    payload = _strict_mapping(
        value,
        {"schema_version", "components", "catalog_sha256"},
        label,
    )
    catalog = ComponentCatalog(
        schema_version=_strict_string(
            payload["schema_version"],
            label + ".schema_version",
        ),
        components=tuple(
            _component_from_mapping(item, f"{label}.components[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["components"], label + ".components")
            )
        ),
        catalog_sha256=_strict_string(
            payload["catalog_sha256"],
            label + ".catalog_sha256",
        ),
    )
    catalog.validate()
    return catalog


def project_inventory_from_mapping(value: Mapping[str, Any]) -> ProjectInventory:
    fields = {
        "schema_version",
        "source_kind",
        "root_name",
        "source_sha256",
        "imported_source_snapshot_sha256",
        "project_snapshot_sha256",
        "manifest",
        "modules",
        "target",
        "metadata",
        "entrypoints",
        "namespaces",
        "dependencies",
        "logical_resource_ids",
        "logical_resource_references",
        "component_catalog",
        "warnings",
        "inventory_sha256",
    }
    payload = _strict_mapping(value, fields, "project inventory")
    inventory = ProjectInventory(
        schema_version=_strict_string(payload["schema_version"], "schema_version"),
        source_kind=_strict_string(payload["source_kind"], "source_kind"),
        root_name=_strict_string(payload["root_name"], "root_name"),
        source_sha256=_strict_string(payload["source_sha256"], "source_sha256"),
        imported_source_snapshot_sha256=_strict_string(
            payload["imported_source_snapshot_sha256"],
            "imported_source_snapshot_sha256",
        ),
        project_snapshot_sha256=_strict_string(
            payload["project_snapshot_sha256"],
            "project_snapshot_sha256",
        ),
        manifest=tuple(
            _evidence_from_mapping(item, f"manifest[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["manifest"], "manifest")
            )
        ),
        modules=tuple(
            _module_from_mapping(item, f"modules[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["modules"], "modules")
            )
        ),
        target=_target_from_mapping(payload["target"], "target"),
        metadata=tuple(
            _metadata_from_mapping(item, f"metadata[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["metadata"], "metadata")
            )
        ),
        entrypoints=tuple(
            _entrypoint_from_mapping(item, f"entrypoints[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["entrypoints"], "entrypoints")
            )
        ),
        namespaces=_string_tuple(payload["namespaces"], "namespaces"),
        dependencies=tuple(
            _dependency_from_mapping(item, f"dependencies[{index}]")
            for index, item in enumerate(
                _strict_sequence(payload["dependencies"], "dependencies")
            )
        ),
        logical_resource_ids=_string_tuple(
            payload["logical_resource_ids"],
            "logical_resource_ids",
        ),
        logical_resource_references=_string_tuple(
            payload["logical_resource_references"],
            "logical_resource_references",
        ),
        component_catalog=_catalog_from_mapping(
            payload["component_catalog"],
            "component_catalog",
        ),
        warnings=_string_tuple(payload["warnings"], "warnings"),
        inventory_sha256=_strict_string(
            payload["inventory_sha256"],
            "inventory_sha256",
        ),
    )
    inventory.validate()
    return inventory


def validate_project_inventory_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on a serialized inventory and return its canonical mapping."""

    return project_inventory_from_mapping(value).to_dict()


__all__ = [
    "project_inventory_from_mapping",
    "validate_project_inventory_payload",
]
