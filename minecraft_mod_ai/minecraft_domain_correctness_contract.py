from __future__ import annotations

"""Fail-closed Minecraft API compatibility guard for deterministic generation.

Legacy deterministic generators contain concrete Minecraft/Fabric API assumptions.
They may mutate a project only when the authoritative ``PlatformAdapter`` explicitly
advertises reviewed deterministic templates for every requested module kind. This
module intentionally owns no Minecraft-version table of its own.
"""

from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

_MARKER = "_mmm_minecraft_domain_correctness_v2"


def _adapter_from_project(project_root: str | Path) -> Any:
    from .platform_catalog import adapter_from_project

    return adapter_from_project(project_root)


def _adapter_for_spec(spec: Any) -> Any:
    from .platform_catalog import adapter_for_lock_values

    return adapter_for_lock_values(spec.platform)


def _raw_advertised_kinds(adapter: Any) -> frozenset[str]:
    advertised = getattr(adapter, "deterministic_module_kinds", frozenset())
    if not isinstance(advertised, (set, frozenset, tuple, list)):
        return frozenset()
    return frozenset(str(kind) for kind in advertised if str(kind).strip())


def _advertised_kinds(adapter: Any, extended_module: Any) -> frozenset[str]:
    generator_supported = frozenset(str(kind) for kind in extended_module._SUPPORTED)
    return generator_supported.intersection(_raw_advertised_kinds(adapter))


def _requested_kinds(payload: Mapping[str, Any]) -> frozenset[str]:
    modules = payload.get("modules")
    if not isinstance(modules, list):
        return frozenset()
    return frozenset(
        str(raw.get("kind"))
        for raw in modules
        if isinstance(raw, Mapping) and isinstance(raw.get("kind"), str)
    )


def _requested_spec_kinds(spec: Any) -> frozenset[str]:
    requested: set[str] = set()
    for content in getattr(spec, "contents", ()) or ():
        kind = getattr(content, "kind", None)
        value = getattr(kind, "value", kind)
        if value is not None and str(value).strip():
            requested.add(str(value))
    if getattr(spec, "boss", None) is not None:
        requested.add("boss")
    return frozenset(requested)


def _target_label(adapter: Any) -> str:
    loader = str(getattr(adapter, "loader", "unknown") or "unknown")
    version = str(getattr(adapter, "minecraft_version", "unknown") or "unknown")
    adapter_id = str(getattr(adapter, "adapter_id", "unknown") or "unknown")
    api_family = str(getattr(adapter, "source_api_family", "unknown") or "unknown")
    return f"{loader} Minecraft {version} adapter={adapter_id} api_family={api_family}"


def _unsupported_reason(
    *,
    allowed: frozenset[str],
    requested: frozenset[str],
) -> str | None:
    if not allowed:
        return "the authoritative platform adapter advertises no reviewed deterministic module templates"
    missing = requested - allowed
    if missing:
        return (
            "requested module kinds are not reviewed for this target: "
            + ", ".join(sorted(missing))
        )
    return None


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
    reason = _unsupported_reason(allowed=allowed, requested=requested)
    if reason is None:
        return
    raise runtime_module.AgentToolRuntimeError(
        "Deterministic Minecraft content generation is unavailable for "
        f"{_target_label(adapter)}: {reason}. No files were changed; use target-grounded "
        "source editing with version-specific evidence instead."
    )


def _guard_generation_spec(spec: Any, *, error_type: type[Exception]) -> None:
    """Reject legacy project generation before its first filesystem mutation."""

    adapter = _adapter_for_spec(spec)
    allowed = _raw_advertised_kinds(adapter)
    requested = _requested_spec_kinds(spec)
    reason = _unsupported_reason(allowed=allowed, requested=requested)
    if reason is None:
        return
    raise error_type(
        "Deterministic Minecraft project generation is unavailable for "
        f"{_target_label(adapter)}: {reason}. No project files were generated; use the "
        "target-grounded scaffold/source path whose API compatibility was verified by "
        "the platform provider."
    )


def _install_execute_guard(contract_module: Any) -> None:
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


def _install_generate_guard(generator_class: type[Any], *, error_type: type[Exception]) -> None:
    current = generator_class.generate
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate(self: Any, spec: Any, root: Path) -> Any:
        _guard_generation_spec(spec, error_type=error_type)
        return current(self, spec, root)

    setattr(generate, _MARKER, True)
    generate.__wrapped__ = current  # type: ignore[attr-defined]
    generator_class.generate = generate


def install(contract_module: Any | None = None) -> None:
    """Install guards on every legacy deterministic generation entry point.

    Passing a contract module is a narrow test/embedding hook and only wraps that
    module. Normal runtime installation wraps both the agent tool and project generator
    classes, including callers that bypass the agent tool entirely.
    """

    if contract_module is not None:
        _install_execute_guard(contract_module)
        return

    from . import deterministic_minecraft_content_contract
    from .generator import FabricProjectGenerator, GenerationError
    from .scalable_generator import ScalableFabricProjectGenerator

    _install_execute_guard(deterministic_minecraft_content_contract)
    _install_generate_guard(FabricProjectGenerator, error_type=GenerationError)
    _install_generate_guard(ScalableFabricProjectGenerator, error_type=GenerationError)


__all__ = [
    "_advertised_kinds",
    "_guard_generation_spec",
    "_guard_target_capability",
    "_requested_kinds",
    "_requested_spec_kinds",
    "install",
]
