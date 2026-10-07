from __future__ import annotations

"""Deterministic preflight for built-in production generation inputs.

The complete orchestrator may execute independent generation nodes concurrently. The
runtime therefore validates the normalized generation set together with persisted
project state immediately before the first generation node is dispatched.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .complete_spec import MODULE_KINDS
from .extended_content_generator import (
    ExtendedContentError,
    validate_extended_module_contract,
)
from .geckolib_generation_contract import (
    GeckoLibGenerationContractError,
    geckolib_entity_inputs_from_module_config,
    preflight_geckolib_generation_target,
    validate_existing_geckolib_records,
)
from .platform_backend_contract import (
    ENTITY_PIPELINE_KINDS as _ENTITY_KINDS,
    EXTENDED_CONTENT_KINDS as _EXTENDED_CONTENT_KINDS,
    SYSTEM_KIND_TO_PACK as _SYSTEM_PACK_BY_KIND,
    deterministic_backend_capabilities,
    missing_production_backend_capabilities,
)
from .production_routing_contract import ProductionRoutingSnapshot
from .scale_policy import ScalePolicy
from .system_pack_validation import validate_system_modules
from .typed_plan_production import validate_typed_plan_generation_contract


class ProductionGenerationPreflightError(ValueError):
    """A normalized module set cannot enter deterministic built-in generation."""


def _system_module_dict(module: Any) -> dict[str, Any]:
    return {
        "module_id": str(module.module_id),
        "kind": str(module.kind),
        "config": module.config,
        "depends_on": list(module.depends_on),
        "required_gates": list(module.required_gates),
    }


def _system_groups(modules: Iterable[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for module in modules:
        pack_id = _SYSTEM_PACK_BY_KIND.get(str(module.kind))
        if pack_id is not None:
            groups.setdefault(pack_id, []).append(module)
    return groups


def _validate_system_group(pack_id: str, modules: list[dict[str, Any]]) -> None:
    try:
        validate_system_modules(pack_id, modules)
    except (TypeError, ValueError) as exc:
        raise ProductionGenerationPreflightError(
            f"System pack {pack_id} cannot enter generation: {exc}"
        ) from exc


def _direct_routed_modules(
    modules: tuple[Any, ...],
    *,
    routing: ProductionRoutingSnapshot | None,
    artifact_owners: Iterable[str],
) -> tuple[Any, ...]:
    if routing is not None:
        routes = routing.route_by_module_id
        missing = [
            str(getattr(module, "module_id", "") or "")
            for module in modules
            if str(getattr(module, "module_id", "") or "") not in routes
        ]
        if missing:
            raise ProductionGenerationPreflightError(
                f"PRODUCTION_ROUTING_MISSING_MODULE: {missing[:20]}"
            )
        return tuple(
            module
            for module in modules
            if routes[str(module.module_id)].owner == "native_host"
        )

    artifact_owned = {
        str(module_id).strip()
        for module_id in artifact_owners
        if str(module_id).strip()
    }
    return tuple(
        module
        for module in modules
        if str(getattr(module, "module_id", "") or "").strip() not in artifact_owned
    )


def validate_production_generation_modules(
    modules: Iterable[Any],
    *,
    policy: ScalePolicy | None = None,
    validate_system_packs: bool = True,
    artifact_owners: Iterable[str] = (),
    routing: ProductionRoutingSnapshot | None = None,
) -> None:
    """Validate normalized built-in module inputs without touching project state."""

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    materialized = tuple(modules)
    direct_routed = _direct_routed_modules(
        materialized,
        routing=routing,
        artifact_owners=artifact_owners,
    )

    for module in direct_routed:
        kind = str(getattr(module, "kind", "") or "").strip()
        config = getattr(module, "config", None)
        if kind not in MODULE_KINDS:
            raise ProductionGenerationPreflightError(
                f"UNSUPPORTED_PRODUCTION_MODULE_KIND: {kind!r}"
            )
        if isinstance(config, dict) and config.get("implementation") is not None:
            raise ProductionGenerationPreflightError(
                "CUSTOM_JAVA_BACKEND_REMOVED: production modules cannot carry "
                f"implementation overrides: {getattr(module, 'module_id', '<unknown>')}"
            )
        if kind in _EXTENDED_CONTENT_KINDS:
            try:
                validate_extended_module_contract(module, policy=policy)
            except ExtendedContentError as exc:
                raise ProductionGenerationPreflightError(
                    f"Extended content module {module.module_id} cannot enter "
                    f"generation: {exc}"
                ) from exc
        if kind in _ENTITY_KINDS:
            try:
                geckolib_entity_inputs_from_module_config(
                    kind,
                    module.config,
                    policy=policy,
                )
            except GeckoLibGenerationContractError as exc:
                raise ProductionGenerationPreflightError(
                    f"Entity module {module.module_id} cannot enter GeckoLib generation: {exc}"
                ) from exc

    if not validate_system_packs:
        return
    for pack_id, group in sorted(_system_groups(direct_routed).items()):
        _validate_system_group(
            pack_id,
            [_system_module_dict(module) for module in group],
        )


def validate_production_generation_project(
    project_root: str | Path,
    modules: Iterable[Any],
    *,
    mod_id: str,
    package_name: str,
    policy: ScalePolicy | None = None,
    artifact_owners: Iterable[str] = (),
    routing: ProductionRoutingSnapshot | None = None,
) -> None:
    """Validate all deterministic state before concurrent generation dispatch begins."""

    policy = policy or ScalePolicy.from_environment()
    materialized = tuple(modules)
    direct_routed = _direct_routed_modules(
        materialized,
        routing=routing,
        artifact_owners=artifact_owners,
    )
    validate_production_generation_modules(
        direct_routed,
        policy=policy,
        validate_system_packs=False,
        routing=routing,
    )

    from .platform_catalog import adapter_from_project

    try:
        adapter = adapter_from_project(project_root)
    except Exception as exc:
        raise ProductionGenerationPreflightError(
            f"Production target receipt is unavailable before generation: {exc}"
        ) from exc
    for module in direct_routed:
        if str(module.kind) == "typed_host":
            try:
                validate_typed_plan_generation_contract(
                    module,
                    package_name=package_name,
                    mod_id=mod_id,
                )
            except (TypeError, ValueError) as exc:
                raise ProductionGenerationPreflightError(
                    f"Typed host module {module.module_id} cannot enter generation: {exc}"
                ) from exc
        available_backend = deterministic_backend_capabilities(adapter)
        if routing is not None:
            route = routing.route_by_module_id[str(module.module_id)]
            missing_backend = frozenset(
                set(route.required_capabilities) - set(available_backend)
            )
        else:
            missing_backend = missing_production_backend_capabilities(
                available_backend,
                str(module.kind),
                module.config,
            )
        if missing_backend:
            raise ProductionGenerationPreflightError(
                "DETERMINISTIC_BACKEND_REQUIRED: "
                f"{module.module_id}/{module.kind} requires missing backend "
                f"capabilities {sorted(missing_backend)}"
            )

    has_entities = any(
        str(module.kind) in _ENTITY_KINDS
        for module in direct_routed
    )
    if has_entities:
        try:
            preflight_geckolib_generation_target(
                project_root,
                mod_id=mod_id,
                package_name=package_name,
            )
            validate_existing_geckolib_records(project_root)
        except GeckoLibGenerationContractError as exc:
            raise ProductionGenerationPreflightError(
                f"GeckoLib project cannot enter entity generation: {exc}"
            ) from exc

    system_groups = _system_groups(direct_routed)
    if not system_groups:
        return
    from .system_pack_generator import iter_system_module_records

    for pack_id, group in sorted(system_groups.items()):
        try:
            existing = iter_system_module_records(
                project_root,
                mod_id=mod_id,
                pack_id=pack_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ProductionGenerationPreflightError(
                f"System pack {pack_id} existing records are invalid: {exc}"
            ) from exc
        merged = {str(item["module_id"]): item for item in existing}
        for module in group:
            item = _system_module_dict(module)
            merged[str(module.module_id)] = item
        _validate_system_group(
            pack_id,
            [merged[module_id] for module_id in sorted(merged)],
        )


__all__ = [
    "ProductionGenerationPreflightError",
    "validate_production_generation_modules",
    "validate_production_generation_project",
]
