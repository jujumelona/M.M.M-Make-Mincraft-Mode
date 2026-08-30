from __future__ import annotations

"""Fail-closed Minecraft API compatibility boundaries for deterministic generation.

Legacy deterministic generators contain concrete Minecraft/Fabric API assumptions.
They may mutate a project only when the authoritative ``PlatformAdapter`` explicitly
advertises reviewed deterministic templates for every requested module kind. This
module intentionally owns no Minecraft-version table of its own.
"""

import sys
from functools import wraps
from pathlib import Path
from typing import Any

_MARKER = "_mmm_minecraft_domain_correctness_v3"


def _adapter_from_project(project_root: str | Path) -> Any:
    from .platform_catalog import adapter_from_project

    return adapter_from_project(project_root)


def _adapter_for_spec(spec: Any) -> Any:
    from .platform_catalog import adapter_for_lock_values

    return adapter_for_lock_values(spec.platform)


def _normalize_kind(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _raw_advertised_kinds(adapter: Any) -> frozenset[str]:
    advertised = getattr(adapter, "deterministic_module_kinds", frozenset())
    if not isinstance(advertised, (set, frozenset, tuple, list)):
        return frozenset()
    return frozenset(
        normalized
        for kind in advertised
        if (normalized := _normalize_kind(kind))
    )


def _supported_kinds(extended_module: Any) -> frozenset[str]:
    return frozenset(
        normalized
        for kind in getattr(extended_module, "_SUPPORTED", ())
        if (normalized := _normalize_kind(kind))
    )


def _advertised_kinds(adapter: Any, extended_module: Any) -> frozenset[str]:
    return _supported_kinds(extended_module).intersection(_raw_advertised_kinds(adapter))


def _requested_module_kinds(
    modules: tuple[Any, ...],
    *,
    supported: frozenset[str],
) -> frozenset[str]:
    requested: set[str] = set()
    for module in modules:
        raw_kind = getattr(module, "kind", None)
        value = getattr(raw_kind, "value", raw_kind)
        kind = _normalize_kind(value)
        if kind in supported:
            requested.add(kind)
    return frozenset(requested)


def _requested_spec_kinds(spec: Any) -> frozenset[str]:
    requested: set[str] = set()
    for content in getattr(spec, "contents", ()) or ():
        raw_kind = getattr(content, "kind", None)
        value = getattr(raw_kind, "value", raw_kind)
        kind = _normalize_kind(value)
        if kind:
            requested.add(kind)
    if getattr(spec, "boss", None) is not None:
        requested.add("boss")
    return frozenset(requested)


def _target_label(adapter: Any) -> str:
    loader = _normalize_kind(getattr(adapter, "loader", None)) or "unknown"
    version = _normalize_kind(getattr(adapter, "minecraft_version", None)) or "unknown"
    adapter_id = _normalize_kind(getattr(adapter, "adapter_id", None)) or "unknown"
    api_family = _normalize_kind(getattr(adapter, "source_api_family", None)) or "unknown"
    return f"{loader} Minecraft {version} adapter={adapter_id} api_family={api_family}"


def _unsupported_reason(
    *,
    allowed: frozenset[str],
    requested: frozenset[str],
    require_reviewed_target: bool,
) -> str | None:
    if require_reviewed_target and not allowed:
        return "the authoritative platform adapter advertises no reviewed deterministic module templates"
    missing = requested - allowed
    if missing:
        return (
            "requested module kinds are not reviewed for this target: "
            + ", ".join(sorted(missing))
        )
    return None


def _guard_generation_spec(spec: Any, *, error_type: type[Exception]) -> None:
    """Reject legacy scaffold/project generation before its first filesystem mutation."""

    adapter = _adapter_for_spec(spec)
    allowed = _raw_advertised_kinds(adapter)
    requested = _requested_spec_kinds(spec)
    reason = _unsupported_reason(
        allowed=allowed,
        requested=requested,
        require_reviewed_target=True,
    )
    if reason is None:
        return
    raise error_type(
        "Deterministic Minecraft project generation is unavailable for "
        f"{_target_label(adapter)}: {reason}. No project files were generated; use the "
        "target-grounded scaffold/source path whose API compatibility was verified by "
        "the platform provider."
    )


def _guard_extended_content(
    project_root: str | Path,
    modules: tuple[Any, ...],
    extended_module: Any,
) -> None:
    """Reject stale deterministic content templates at the shared mutation primitive."""

    supported = _supported_kinds(extended_module)
    requested = _requested_module_kinds(modules, supported=supported)
    if not requested:
        return
    adapter = _adapter_from_project(project_root)
    allowed = supported.intersection(_raw_advertised_kinds(adapter))
    reason = _unsupported_reason(
        allowed=allowed,
        requested=requested,
        require_reviewed_target=True,
    )
    if reason is None:
        return
    raise extended_module.ExtendedContentError(
        "Deterministic Minecraft content generation is unavailable for "
        f"{_target_label(adapter)}: {reason}. No files were changed; use target-grounded "
        "source editing with version-specific evidence instead."
    )


def _callable_lineage(callable_obj: Any) -> tuple[Any, ...]:
    """Return one wrapper chain by identity, protecting against malformed cycles."""

    lineage: list[Any] = []
    seen: set[int] = set()
    current = callable_obj
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        lineage.append(current)
        current = getattr(current, "__wrapped__", None)
    return tuple(lineage)


def _retarget_package_aliases(
    name: str,
    stale_callables: tuple[Any, ...],
    replacement: Any,
) -> None:
    """Retarget import-by-value aliases that captured any prior wrapper layer."""

    stale_ids = {id(callable_obj) for callable_obj in stale_callables}
    for module_name, loaded in tuple(sys.modules.items()):
        if not (
            module_name == "minecraft_mod_ai"
            or module_name.startswith("minecraft_mod_ai.")
        ):
            continue
        if loaded is None:
            continue
        try:
            namespace = vars(loaded)
        except TypeError:
            continue
        candidate = namespace.get(name)
        if callable(candidate) and id(candidate) in stale_ids:
            namespace[name] = replacement


def _install_extended_content_guard(extended_module: Any) -> Any:
    current = extended_module.generate_extended_content
    if getattr(current, _MARKER, False):
        return current

    stale_lineage = _callable_lineage(current)

    @wraps(current)
    def generate_extended_content(*args: Any, **kwargs: Any) -> dict[str, Any]:
        project_root = kwargs.get("project_root")
        raw_modules = kwargs.get("modules")
        if project_root is None or raw_modules is None:
            return current(*args, **kwargs)

        modules = tuple(raw_modules)
        _guard_extended_content(project_root, modules, extended_module)
        forwarded = dict(kwargs)
        forwarded["modules"] = modules
        return current(*args, **forwarded)

    setattr(generate_extended_content, _MARKER, True)
    generate_extended_content.__wrapped__ = current  # type: ignore[attr-defined]
    extended_module.generate_extended_content = generate_extended_content
    _retarget_package_aliases(
        "generate_extended_content",
        stale_lineage,
        generate_extended_content,
    )
    return generate_extended_content


def _install_generate_guard(generator_class: type[Any], *, error_type: type[Exception]) -> Any:
    current = generator_class.generate
    if getattr(current, _MARKER, False):
        return current

    @wraps(current)
    def generate(self: Any, spec: Any, root: Path) -> Any:
        _guard_generation_spec(spec, error_type=error_type)
        return current(self, spec, root)

    setattr(generate, _MARKER, True)
    generate.__wrapped__ = current  # type: ignore[attr-defined]
    generator_class.generate = generate
    return generate


def install() -> None:
    """Install the minimum complete fail-closed boundary set.

    The shared extended-content mutation primitive is guarded once, so agent tools,
    complete orchestration, scalable generation and direct callers share one capability
    check. Project-generator guards remain separate because their scaffold and GameTest
    writes occur before the shared content primitive.
    """

    from . import extended_content_generator
    from .generator import FabricProjectGenerator, GenerationError
    from .scalable_generator import ScalableFabricProjectGenerator

    _install_extended_content_guard(extended_content_generator)
    _install_generate_guard(FabricProjectGenerator, error_type=GenerationError)
    _install_generate_guard(ScalableFabricProjectGenerator, error_type=GenerationError)


__all__ = [
    "_advertised_kinds",
    "_callable_lineage",
    "_guard_extended_content",
    "_guard_generation_spec",
    "_install_extended_content_guard",
    "_install_generate_guard",
    "_raw_advertised_kinds",
    "_requested_module_kinds",
    "_requested_spec_kinds",
    "_retarget_package_aliases",
    "install",
]
