from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .project_edit import (
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
from .system_templates_class_skill import _class_skill_java
from .system_templates_common import (
    _config_loader_java,
    _persistent_store_java,
)
from .system_templates_economy import _economy_java
from .system_templates_quest import _quest_java
from .system_templates_social import _gui_java, _party_java

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_PACKS = frozenset(
    {
        "quest-system",
        "class-skill-system",
        "economy-shop",
        "gui-networking",
        "party-guild",
    }
)
_PACK_KINDS = {
    "quest-system": {"quest"},
    "class-skill-system": {"class", "skill"},
    "economy-shop": {"economy", "shop"},
    "gui-networking": {"gui", "networking"},
    "party-guild": {"party", "guild"},
}
_QUEST_OBJECTIVES = {"kill", "break", "manual"}


def generate_system_pack(
    *,
    project_root: str | Path,
    pack_id: str,
    mod_id: str,
    package_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate validated data-driven Fabric gameplay systems.

    Built-in templates only accept semantics they actually implement. Unsupported
    semantics must be emitted as ``custom_java`` or set ``implementation=custom`` so
    the indexed custom generator writes the required source rather than silently
    reducing the feature.
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
    _validate_modules(pack_id, modules)

    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise ValueError("System pack target does not match fabric.mod.json.")

    class_name = "".join(part.capitalize() for part in pack_id.split("-"))
    if not class_name.endswith("System"):
        class_name += "System"
    package_path = package_name.replace(".", "/")
    relative_java = (
        f"src/main/java/{package_path}/system/{class_name}.java"
    )
    shared_store = (
        f"src/main/java/{package_path}/system/MmmPersistentStore.java"
    )
    shared_config = (
        f"src/main/java/{package_path}/system/MmmSystemConfig.java"
    )
    contract_relative = (
        f"src/main/resources/data/{mod_id}/mmm_systems/{pack_id}.json"
    )
    contract = {
        "schema_version": f"mmm/{pack_id}-v4",
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
    write_receipt = write_text_files(
        info,
        files,
        replace_existing=True,
    )
    bind_receipt = ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.system.{class_name}",
        call_line=f"{class_name}.register()",
        marker=f"system:{pack_id}",
    )
    return {
        "schema_version": "mmm/system-pack-generation-v4",
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
                else "client GUI launch test"
            ),
            "multiplayer authority and replay test",
        ],
    }


def _validate_modules(pack_id: str, modules: list[Any]) -> None:
    expected_kinds = _PACK_KINDS[pack_id]
    seen: set[str] = set()
    for item in modules:
        if not isinstance(item, dict):
            raise ValueError("System module must be an object.")
        allowed_fields = {
            "module_id",
            "kind",
            "config",
            "depends_on",
            "required_gates",
        }
        if set(item) != allowed_fields:
            raise ValueError(
                f"System module fields are invalid: {sorted(set(item))}"
            )
        module_id = str(item["module_id"])
        if not _ID.fullmatch(module_id) or module_id in seen:
            raise ValueError(
                f"Invalid or duplicate system module id: {module_id!r}"
            )
        seen.add(module_id)
        kind = str(item["kind"])
        if kind not in expected_kinds:
            raise ValueError(
                f"System pack {pack_id} cannot contain kind {kind!r}."
            )
        config = item["config"]
        if not isinstance(config, dict):
            raise ValueError(
                f"System module config must be an object: {module_id}"
            )
        if config.get("implementation") == "custom":
            raise ValueError(
                f"Custom module {module_id} must not be sent to built-in system pack."
            )
        if kind == "quest":
            _validate_quest(module_id, config)
        elif kind == "class":
            _validate_class(module_id, config)
        elif kind == "skill":
            _validate_skill(module_id, config)
        elif kind == "economy":
            _validate_economy(module_id, config)
        elif kind == "shop":
            _validate_shop(module_id, config)
        elif kind in {"gui", "networking"}:
            _validate_gui(module_id, config)
        elif kind in {"party", "guild"}:
            _validate_social(module_id, config)


def _validate_quest(module_id: str, config: dict[str, Any]) -> None:
    objective = str(config.get("objective", "manual"))
    if objective not in _QUEST_OBJECTIVES:
        raise ValueError(
            f"Quest {module_id} objective {objective!r} is not a built-in objective; use custom_java."
        )
    target = str(config.get("target", module_id))
    if objective in {"kill", "break"} and not _RESOURCE_ID.fullmatch(target):
        raise ValueError(
            f"Quest {module_id} requires a namespaced target for {objective}."
        )
    _positive_int(config.get("required", 1), f"{module_id}.required")
    reward_item = str(config.get("reward_item", ""))
    if reward_item and not _RESOURCE_ID.fullmatch(reward_item):
        raise ValueError(
            f"Quest {module_id} reward_item must be namespaced."
        )
    _positive_int(
        config.get("reward_count", 1),
        f"{module_id}.reward_count",
    )
    _finite_number(
        config.get("reward_currency", 0.0),
        f"{module_id}.reward_currency",
    )


def _validate_class(module_id: str, config: dict[str, Any]) -> None:
    display = str(config.get("display_name", module_id)).strip()
    if not display:
        raise ValueError(f"Class {module_id} display_name is empty.")


def _validate_skill(module_id: str, config: dict[str, Any]) -> None:
    effect = str(config.get("effect", "minecraft:speed"))
    if not _RESOURCE_ID.fullmatch(effect):
        raise ValueError(f"Skill {module_id} effect must be namespaced.")
    required_class = str(config.get("required_class", ""))
    if required_class and not _ID.fullmatch(required_class):
        raise ValueError(
            f"Skill {module_id} required_class is invalid."
        )
    _positive_int(
        config.get("duration_ticks", 100),
        f"{module_id}.duration_ticks",
    )
    _nonnegative_int(
        config.get("amplifier", 0),
        f"{module_id}.amplifier",
    )
    _positive_int(
        config.get("cooldown_ticks", 100),
        f"{module_id}.cooldown_ticks",
    )


def _validate_economy(module_id: str, config: dict[str, Any]) -> None:
    value = _finite_number(
        config.get("initial_balance", 0.0),
        f"{module_id}.initial_balance",
    )
    if value < 0:
        raise ValueError(
            f"Economy {module_id} initial_balance must be nonnegative."
        )


def _validate_shop(module_id: str, config: dict[str, Any]) -> None:
    entries = config.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Shop {module_id} requires a non-empty entries list."
        )
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "id",
            "item",
            "count",
            "price",
        }:
            raise ValueError(
                f"Shop {module_id} contains an invalid entry."
            )
        if "id" not in entry or "item" not in entry or "price" not in entry:
            raise ValueError(
                f"Shop {module_id} entry requires id, item and price."
            )
        entry_id = str(entry["id"])
        if not _ID.fullmatch(entry_id) or entry_id in seen:
            raise ValueError(
                f"Shop {module_id} has invalid or duplicate entry {entry_id!r}."
            )
        seen.add(entry_id)
        if not _RESOURCE_ID.fullmatch(str(entry["item"])):
            raise ValueError(
                f"Shop {module_id}/{entry_id} item must be namespaced."
            )
        _positive_int(
            entry.get("count", 1),
            f"{module_id}.{entry_id}.count",
        )
        price = _finite_number(
            entry["price"],
            f"{module_id}.{entry_id}.price",
        )
        if price < 0:
            raise ValueError(
                f"Shop {module_id}/{entry_id} price must be nonnegative."
            )


def _validate_gui(module_id: str, config: dict[str, Any]) -> None:
    actions = config.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError(f"GUI {module_id} actions must be a list.")
    if len(set(str(value) for value in actions)) != len(actions):
        raise ValueError(f"GUI {module_id} actions must be unique.")
    for action in actions:
        if not isinstance(action, str) or not _ID.fullmatch(action):
            raise ValueError(
                f"GUI {module_id} action is invalid: {action!r}"
            )


def _validate_social(module_id: str, config: dict[str, Any]) -> None:
    if config:
        allowed = {"display_name"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"Social module {module_id} has unsupported built-in fields {sorted(unknown)}; use custom_java."
            )


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")
    return value


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite JSON number.")
    return float(value)


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
    return _party_java(package_name, class_name)
