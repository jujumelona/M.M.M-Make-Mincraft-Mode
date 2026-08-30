from __future__ import annotations

"""Fail-closed Minecraft API compatibility guard for deterministic generation.

The legacy deterministic content generator contains concrete Minecraft/Fabric API
assumptions. It may mutate a project only when the authoritative PlatformAdapter
explicitly advertises reviewed deterministic templates for every requested module kind.
This module intentionally owns no version table of its own.
"""

from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

_MARKER = "_mmm_minecraft_domain_correctness_v1"


def _adapter_from_project(project_root: str | Path) -> Any:
    from .platform_catalog import adapter_from_project

    return adapter_from_project(project_root)


def _advertised_kinds(adapter: Any, extended_module: Any) -> frozenset[str]:
    advertised = getattr(adapter, "deterministic_module_kinds", frozenset())
    if not isinstance(advertised, (set, frozenset, tuple, list)):
        return frozenset()
    generator_supported = frozenset(str(kind) for kind in extended_module._SUPPORTED)
    return generator_supported.intersection(str(kind) for kind in advertised)


def _requested_kinds(payload: Mapping[str, Any]) -> frozenset[str]:
    modules = payload.get("modules")
    if not isinstance(modules, list):
        return frozenset()
    return frozenset(
        str(raw.get("kind"))
        for raw in modules
        if isinstance(raw, Mapping) and isinstance(raw.get("kind"), str)
    )


def _target_label(adapter: Any) -> str:
    loader = str(getattr(adapter, "loader", "unknown") or "unknown")
    version = str(getattr(adapter, "minecraft_version", "unknown") or "unknown")
    adapter_id = str(getattr(adapter, "adapter_id", "unknown") or "unknown")
    api_family = str(getattr(adapter, "source_api_family", "unknown") or "unknown")
    return f"{loader} Minecraft {version} adapter={adapter_id} api_family={api_family}"


def _guard_target_capability(
    runtime_module: Any,
    extended_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
) -> None:
    project_root, _project_argument = runtime_module._discover_model_project_root(
        workspace_root
    )
    adapter = _adapter_from_project(project_root)
    allowed = _advertised_kinds(adapter, extended_module)
    requested = _requested_kinds(payload)
    missing = requested - allowed
    if allowed and not missing:
        return

    if not allowed:
        reason = "the authoritative platform adapter advertises no reviewed deterministic module templates"
    else:
        reason = (
            "requested module kinds are not reviewed for this target: "
            + ", ".join(sorted(missing))
        )
    raise runtime_module.AgentToolRuntimeError(
        "Deterministic Minecraft content generation is unavailable for "
        f"{_target_label(adapter)}: {reason}. No files were changed; use target-grounded "
        "source editing with version-specific evidence instead."
    )


def install(contract_module: Any | None = None) -> None:
    if contract_module is None:
        from . import deterministic_minecraft_content_contract as contract_module

    current = contract_module._execute
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def execute(
        runtime_module: Any,
        extended_module: Any,
        workspace_root: str | Path,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        _guard_target_capability(
            runtime_module,
            extended_module,
            workspace_root,
            payload,
        )
        return current(runtime_module, extended_module, workspace_root, payload)

    setattr(execute, _MARKER, True)
    execute.__wrapped__ = current  # type: ignore[attr-defined]
    contract_module._execute = execute


__all__ = [
    "_advertised_kinds",
    "_guard_target_capability",
    "_requested_kinds",
    "install",
]
