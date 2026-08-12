from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_from_project


def _require_fabric_1201(project_root: str | Path, feature: str) -> None:
    adapter = adapter_from_project(project_root)
    if adapter.source_api_family == "fabric_1201":
        return
    raise ValueError(
        f"{feature} deterministic templates are reviewed only for fabric_1201; "
        f"target {adapter.minecraft_version}/{adapter.loader} ({adapter.adapter_id}) "
        "must be implemented through the target-aware custom_java/RAG compiler."
    )


def install(
    *,
    system_module: Any,
    geckolib_module: Any,
    orchestrator_module: Any | None = None,
) -> None:
    """Prevent 1.20.1-only templates from leaking into newer Fabric projects."""

    current_system = system_module.generate_system_pack
    if not getattr(current_system, "_mmm_platform_specialized_guard", False):
        @wraps(current_system)
        def generate_system_pack(*args: Any, **kwargs: Any):
            project_root = kwargs.get("project_root")
            if project_root is None and args:
                project_root = args[0]
            if project_root is None:
                raise ValueError("project_root is required for system-pack generation.")
            _require_fabric_1201(project_root, "Built-in system-pack")
            return current_system(*args, **kwargs)

        generate_system_pack._mmm_platform_specialized_guard = True
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
            _require_fabric_1201(project_root, "Built-in GeckoLib entity")
            return current_gecko(*args, **kwargs)

        generate_geckolib_entity_assets._mmm_platform_specialized_guard = True
        geckolib_module.generate_geckolib_entity_assets = generate_geckolib_entity_assets
        if orchestrator_module is not None:
            orchestrator_module.generate_geckolib_entity_assets = generate_geckolib_entity_assets
