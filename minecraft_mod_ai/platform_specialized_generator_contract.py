from __future__ import annotations

import json
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_from_project

_SYSTEM_INCREMENTAL_STATE: ContextVar[tuple[frozenset[str], bool] | None] = ContextVar(
    "mmm_system_incremental_state",
    default=None,
)


def _require_deterministic_capability(
    project_root: str | Path,
    *,
    required_kinds: frozenset[str],
    feature: str,
) -> None:
    adapter = adapter_from_project(project_root)
    if required_kinds and required_kinds.issubset(adapter.deterministic_module_kinds):
        return
    raise ValueError(
        f"{feature} deterministic templates are not declared by provider "
        f"{adapter.adapter_id} for {adapter.minecraft_version}/{adapter.loader}; "
        "route this work through target-aware custom_java/RAG generation."
    )


_SYSTEM_PACK_KINDS = {
    "quest-system": frozenset({"quest"}),
    "class-skill-system": frozenset({"class", "skill"}),
    "economy-shop": frozenset({"economy", "shop"}),
    "gui-networking": frozenset({"gui", "networking"}),
    "party-guild": frozenset({"party", "guild"}),
}


def _install_incremental_system_records(system_module: Any) -> None:
    current = system_module._system_contract_files
    if getattr(current, "_mmm_incremental_record_writes", False):
        return

    @wraps(current)
    def contract_files(*args: Any, **kwargs: Any):
        files, logical_shards = current(*args, **kwargs)
        state = _SYSTEM_INCREMENTAL_STATE.get()
        if state is None:
            return files, logical_shards
        changed_ids, directory_exists = state
        if not directory_exists:
            return files, logical_shards

        pack_id = kwargs.get("pack_id")
        mod_id = kwargs.get("mod_id")
        if pack_id is None and args:
            pack_id = args[0]
        if mod_id is None and len(args) > 1:
            mod_id = args[1]
        prefix = f"src/main/resources/data/{mod_id}/mmm_systems/{pack_id}/records/"
        filtered = {
            path: content
            for path, content in files.items()
            if not path.startswith(prefix)
            or Path(path).stem in changed_ids
        }
        return filtered, logical_shards

    contract_files._mmm_incremental_record_writes = True
    contract_files.__wrapped__ = current
    system_module._system_contract_files = contract_files


def install(
    *,
    system_module: Any,
    geckolib_module: Any,
    orchestrator_module: Any | None = None,
) -> None:
    """Install platform-stage deterministic generation safety and bounded reuse."""

    _install_incremental_system_records(system_module)

    current_system = system_module.generate_system_pack
    if not getattr(current_system, "_mmm_platform_specialized_guard", False):
        @wraps(current_system)
        def generate_system_pack(*args: Any, **kwargs: Any):
            project_root = kwargs.get("project_root")
            pack_id = kwargs.get("pack_id")
            mod_id = kwargs.get("mod_id")
            config = kwargs.get("config")
            if project_root is None and args:
                project_root = args[0]
            if pack_id is None and len(args) > 1:
                pack_id = args[1]
            if mod_id is None and len(args) > 2:
                mod_id = args[2]
            if config is None and len(args) > 4:
                config = args[4]
            if project_root is None:
                raise ValueError("project_root is required for system-pack generation.")
            required_kinds = _SYSTEM_PACK_KINDS.get(str(pack_id), frozenset())
            _require_deterministic_capability(
                project_root, required_kinds=required_kinds, feature="Built-in system-pack"
            )

            root = Path(project_root).expanduser().resolve()
            contract = (
                root
                / "src/main/resources/data"
                / str(mod_id)
                / "mmm_systems"
                / f"{pack_id}.json"
            )
            directory_exists = False
            if contract.is_file() and not contract.is_symlink():
                try:
                    payload = json.loads(contract.read_text(encoding="utf-8"))
                    directory_exists = (
                        isinstance(payload, dict)
                        and payload.get("storage_schema_version")
                        == system_module._DIRECTORY_SCHEMA
                    )
                except (OSError, json.JSONDecodeError):
                    directory_exists = False

            changed_ids: frozenset[str] = frozenset()
            if isinstance(config, dict) and isinstance(config.get("modules"), list):
                changed_ids = frozenset(
                    str(item.get("module_id", ""))
                    for item in config["modules"]
                    if isinstance(item, dict) and item.get("module_id")
                )
            token = _SYSTEM_INCREMENTAL_STATE.set((changed_ids, directory_exists))
            try:
                return current_system(*args, **kwargs)
            finally:
                _SYSTEM_INCREMENTAL_STATE.reset(token)

        generate_system_pack._mmm_platform_specialized_guard = True
        generate_system_pack._mmm_incremental_record_writes = True
        system_module.generate_system_pack = generate_system_pack
        if orchestrator_module is not None:
            orchestrator_module.generate_system_pack = generate_system_pack

    current_gecko = geckolib_module.generate_geckolib_entity_assets
    if not getattr(current_gecko, "_mmm_platform_specialized_guard", False):
        @wraps(current_gecko)
        def generate_geckolib_entity_assets(*args: Any, **kwargs: Any):
            project_root = kwargs.get("project_root")
            if project_root is None and args:
                project_root = args[0]
            if project_root is None:
                raise ValueError("project_root is required for GeckoLib generation.")
            _require_deterministic_capability(
                project_root, required_kinds=frozenset({"entity"}), feature="Built-in GeckoLib entity"
            )
            return current_gecko(*args, **kwargs)

        generate_geckolib_entity_assets._mmm_platform_specialized_guard = True
        geckolib_module.generate_geckolib_entity_assets = generate_geckolib_entity_assets
        if orchestrator_module is not None:
            orchestrator_module.generate_geckolib_entity_assets = generate_geckolib_entity_assets
