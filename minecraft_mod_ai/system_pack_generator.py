from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .project_edit import (
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
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
) -> dict[str, Any]:
    """Generate only semantics implemented by the built-in Fabric templates.

    All module validation is centralized in ``system_pack_validation`` so the
    accepted JSON contract and generated Java cannot drift apart. Unsupported
    semantics must be routed to indexed custom generation instead of being
    silently reduced.
    """

    if pack_id not in _PACKS:
        raise ValueError(f"Unknown system pack: {pack_id}")
    if not _ID.fullmatch(mod_id) or not _PACKAGE.fullmatch(package_name):
        raise ValueError("Invalid mod id or Java package.")
    if not isinstance(config, dict) or set(config) != {"modules"}:
        raise ValueError("System pack config must contain exactly modules.")
    modules = config["modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("System pack config.modules must be a non-empty list.")
    validate_system_modules(pack_id, modules)

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise ValueError("System pack target does not match fabric.mod.json.")

    class_name = "".join(part.capitalize() for part in pack_id.split("-"))
    if not class_name.endswith("System"):
        class_name += "System"
    package_path = package_name.replace(".", "/")
    relative_java = f"src/main/java/{package_path}/system/{class_name}.java"
    shared_store = f"src/main/java/{package_path}/system/MmmPersistentStore.java"
    shared_config = f"src/main/java/{package_path}/system/MmmSystemConfig.java"
    contract_relative = f"src/main/resources/data/{mod_id}/mmm_systems/{pack_id}.json"
    contract = {
        "schema_version": f"mmm/{pack_id}-v5",
        "pack_id": pack_id,
        "modules": modules,
        "server_authoritative": True,
        "persistent": pack_id != "gui-networking",
        "minecraft_version": "1.20.1",
        "loader": "fabric",
    }
    files = {
        shared_store: _persistent_store_java(package_name, mod_id),
        shared_config: _config_loader_java(package_name),
        relative_java: _system_java(
            pack_id,
            package_name,
            mod_id,
            class_name,
            contract_relative,
        ),
        contract_relative: json.dumps(
            contract,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    }
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
        "definition_count": len(modules),
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
