from __future__ import annotations

import json
import re
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .project_edit import (
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
from .scale_policy import ScalePolicy
from .system_pack_validation import validate_system_modules
from .system_templates_class_skill import _class_skill_java
from .system_templates_common import (
    _config_loader_java,
    _persistent_store_java,
)
from .system_templates_economy import _economy_java
from .system_templates_groups import _party_java
from .system_templates_quest import _quest_java
from .system_templates_social import _gui_java

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_DIRECTORY_SCHEMA = "mmm/system-pack-directory-v1"
_RECORD_SCHEMA = "mmm/system-module-record-v1"
_LEGACY_INDEX_SCHEMA = "mmm/system-pack-index-v1"
_LEGACY_SHARD_SCHEMA = "mmm/system-module-shard-v1"
_LEGACY_NODE_SCHEMA = "mmm/system-module-index-node-v1"
_PACKS = frozenset(
    {
        "quest-system",
        "class-skill-system",
        "economy-shop",
        "gui-networking",
        "party-guild",
    }
)


def supported_system_packs() -> tuple[str, ...]:
    """Return the stable built-in system-pack IDs in deterministic order."""

    return tuple(sorted(_PACKS))


def generate_system_pack(
    *,
    project_root: str | Path,
    pack_id: str,
    mod_id: str,
    package_name: str,
    config: dict[str, Any],
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    """Generate only semantics implemented by the built-in Fabric templates.

    All module validation is centralized in ``system_pack_validation`` so the
    accepted JSON contract and generated Java cannot drift apart. Unsupported
    semantics must be routed to indexed custom generation instead of being
    silently reduced.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if pack_id not in _PACKS:
        raise ValueError(f"Unknown system pack: {pack_id}")
    if not _ID.fullmatch(mod_id) or not _PACKAGE.fullmatch(package_name):
        raise ValueError("Invalid mod id or Java package.")
    if not isinstance(config, dict) or set(config) != {"modules"}:
        raise ValueError("System pack config must contain exactly modules.")
    modules = config["modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("System pack config.modules must be a non-empty list.")

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise ValueError("System pack target does not match fabric.mod.json.")
    merged_by_id = {
        str(item["module_id"]): item
        for item in iter_system_module_records(
            info.root,
            mod_id=mod_id,
            pack_id=pack_id,
        )
    }
    for item in modules:
        if not isinstance(item, dict):
            raise ValueError("System module must be an object.")
        module_id = item.get("module_id")
        if not isinstance(module_id, str):
            raise ValueError("System module id must be a string.")
        merged_by_id[module_id] = item
    merged_modules = [
        merged_by_id[module_id]
        for module_id in sorted(merged_by_id)
    ]
    validate_system_modules(pack_id, merged_modules)

    class_name = "".join(part.capitalize() for part in pack_id.split("-"))
    if not class_name.endswith("System"):
        class_name += "System"
    package_path = package_name.replace(".", "/")
    relative_java = f"src/main/java/{package_path}/system/{class_name}.java"
    shared_store = f"src/main/java/{package_path}/system/MmmPersistentStore.java"
    shared_config = f"src/main/java/{package_path}/system/MmmSystemConfig.java"
    contract_relative = f"src/main/resources/data/{mod_id}/mmm_systems/{pack_id}.json"
    files = {
        shared_store: _persistent_store_java(package_name, mod_id),
        shared_config: _config_loader_java(package_name, mod_id),
        relative_java: _system_java(
            pack_id,
            package_name,
            mod_id,
            class_name,
            contract_relative,
        ),
    }
    contract_files, contract_shards = _system_contract_files(
        pack_id=pack_id,
        mod_id=mod_id,
        modules=merged_modules,
        shard_size=policy.java_shard_size,
        contract_relative=contract_relative,
    )
    files.update(contract_files)
    write_receipt = write_text_files(info, files, replace_existing=True)
    bind_receipt = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.system.{class_name}",
        call_line=f"{class_name}.register()",
        marker=f"system:{pack_id}",
    )
    return {
        "schema_version": "mmm/system-pack-generation-v5",
        "pack_id": pack_id,
        "input_definition_count": len(modules),
        "definition_count": len(merged_modules),
        "definition_record_count": len(merged_modules),
        "definition_shard_count": contract_shards,
        "definition_shard_size": policy.java_shard_size,
        "files": [str(info.root / path) for path in files],
        "write_receipt": write_receipt,
        "binding_receipt": bind_receipt,
        "status": "fabric_binding_generated",
        "required_gates": [
            "JDT diagnostics",
            "Gradle clean build",
            "GameTest",
            (
                "restart persistence test"
                if pack_id != "gui-networking"
                else "client GUI and validated-network-action test"
            ),
            "multiplayer authority and replay test",
        ],
    }


def _system_contract_files(
    *,
    pack_id: str,
    mod_id: str,
    modules: list[dict[str, Any]],
    shard_size: int,
    contract_relative: str,
) -> tuple[dict[str, str], int]:
    relative_base = (
        f"src/main/resources/data/{mod_id}/mmm_systems/{pack_id}"
    )
    resource_base = f"/data/{mod_id}/mmm_systems/{pack_id}"
    files: dict[str, str] = {}
    records_relative = f"{relative_base}/records"
    records_resource = f"{resource_base}/records"
    for module in modules:
        module_id = str(module["module_id"])
        relative = f"{records_relative}/{module_id}.json"
        files[relative] = _json_text(
            {
                "schema_version": _RECORD_SCHEMA,
                "module": module,
            }
        )

    files[contract_relative] = _json_text(
        {
            "schema_version": f"mmm/{pack_id}-v5",
            "storage_schema_version": _DIRECTORY_SCHEMA,
            "pack_id": pack_id,
            "module_count": len(modules),
            "directory": records_resource,
            "server_authoritative": True,
            "persistent": pack_id != "gui-networking",
            "minecraft_version": "1.20.1",
            "loader": "fabric",
        }
    )
    logical_shards = (len(modules) + shard_size - 1) // shard_size
    return files, logical_shards


def iter_system_module_records(
    project_root: str | Path,
    *,
    mod_id: str,
    pack_id: str,
) -> tuple[dict[str, Any], ...]:
    """Read both the stable record directory and legacy sharded catalogs."""

    root = Path(project_root).resolve()
    resources = root / "src/main/resources"
    contract = resources / f"data/{mod_id}/mmm_systems/{pack_id}.json"
    if not contract.exists():
        return ()
    if not contract.is_file() or contract.is_symlink():
        raise ValueError("System pack contract is not a regular file.")
    try:
        catalog = json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("System pack contract is not valid JSON.") from exc
    if not isinstance(catalog, dict):
        raise ValueError("System pack contract must be an object.")
    if catalog.get("pack_id") not in {None, pack_id}:
        raise ValueError("System pack contract pack_id does not match its path.")

    inline = catalog.get("modules")
    if isinstance(inline, list):
        modules = inline
    else:
        storage = catalog.get("storage_schema_version")
        if storage == _DIRECTORY_SCHEMA:
            modules = _read_directory_records(
                resources,
                catalog,
            )
        elif storage == _LEGACY_INDEX_SCHEMA:
            modules = _read_legacy_catalog(
                resources,
                catalog,
            )
        else:
            raise ValueError("Unsupported system pack storage schema.")

    expected = catalog.get("module_count", len(modules))
    if type(expected) is not int or expected != len(modules):
        raise ValueError("System pack module count does not match its records.")
    by_id: dict[str, dict[str, Any]] = {}
    for item in modules:
        if not isinstance(item, dict):
            raise ValueError("System pack contains a non-object module.")
        module_id = item.get("module_id")
        if not isinstance(module_id, str) or not _ID.fullmatch(module_id):
            raise ValueError("System pack contains an invalid module id.")
        if module_id in by_id:
            raise ValueError(
                f"System pack contains duplicate module id: {module_id}"
            )
        by_id[module_id] = item
    return tuple(by_id[module_id] for module_id in sorted(by_id))


def _read_directory_records(
    resources: Path,
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    directory_raw = catalog.get("directory")
    directory = _resource_target(
        resources,
        directory_raw,
        expect_directory=True,
    )
    modules: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if not path.is_file() or path.is_symlink():
            raise ValueError("System module record is not a regular file.")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"System module record is not valid JSON: {path.name}"
            ) from exc
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != _RECORD_SCHEMA
            or not isinstance(record.get("module"), dict)
        ):
            raise ValueError(
                f"System module record has an invalid schema: {path.name}"
            )
        module = record["module"]
        if path.stem != module.get("module_id"):
            raise ValueError(
                f"System module record filename does not match its id: {path.name}"
            )
        modules.append(module)
    return modules


def _read_legacy_catalog(
    resources: Path,
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    root_resource = catalog.get("root")
    pending = [root_resource]
    visited: set[str] = set()
    modules: list[dict[str, Any]] = []
    while pending:
        resource = pending.pop()
        if not isinstance(resource, str) or resource in visited:
            raise ValueError("Legacy system pack catalog contains a cycle.")
        visited.add(resource)
        path = _resource_target(resources, resource)
        try:
            node = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Legacy system catalog node is invalid: {resource}"
            ) from exc
        if not isinstance(node, dict):
            raise ValueError("Legacy system catalog node must be an object.")
        schema = node.get("schema_version")
        if schema == _LEGACY_SHARD_SCHEMA:
            shard = node.get("modules")
            if not isinstance(shard, list):
                raise ValueError("Legacy system module shard is invalid.")
            modules.extend(shard)
            continue
        if schema != _LEGACY_NODE_SCHEMA:
            raise ValueError("Unsupported legacy system catalog node.")
        children = node.get("children")
        if not isinstance(children, list) or not children:
            raise ValueError("Legacy system catalog index is empty.")
        if not all(isinstance(item, str) for item in children):
            raise ValueError("Legacy system catalog child path is invalid.")
        pending.extend(reversed(children))
    return modules


def _resource_target(
    resources: Path,
    resource: Any,
    *,
    expect_directory: bool = False,
) -> Path:
    if (
        not isinstance(resource, str)
        or not resource.startswith("/data/")
        or "\\" in resource
    ):
        raise ValueError("System catalog resource path is invalid.")
    relative = PurePosixPath(resource[1:])
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("System catalog resource path is unsafe.")
    candidate = resources
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("System catalog resource path is a symlink.")
    target = candidate.resolve()
    try:
        target.relative_to(resources.resolve())
    except ValueError as exc:
        raise ValueError("System catalog resource path escaped resources.") from exc
    valid = target.is_dir() if expect_directory else target.is_file()
    if not valid:
        kind = "directory" if expect_directory else "file"
        raise ValueError(f"System catalog {kind} is missing: {resource}")
    return target


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _system_java(
    pack_id: str,
    package_name: str,
    mod_id: str,
    class_name: str,
    resource_path: str,
) -> str:
    absolute_resource = "/" + resource_path.replace("\\", "/").removeprefix(
        "src/main/resources/"
    )
    if pack_id == "quest-system":
        return _quest_java(package_name, class_name, absolute_resource)
    if pack_id == "class-skill-system":
        return _class_skill_java(package_name, class_name, absolute_resource)
    if pack_id == "economy-shop":
        return _economy_java(package_name, class_name, absolute_resource)
    if pack_id == "gui-networking":
        return _gui_java(
            package_name,
            mod_id,
            class_name,
            absolute_resource,
        )
    return _party_java(package_name, class_name, absolute_resource)
