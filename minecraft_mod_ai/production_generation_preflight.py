from __future__ import annotations

"""Deterministic preflight for built-in production generation inputs.

The complete orchestrator may execute independent generation nodes concurrently. Any
configuration error knowable from the approved proposal is rejected during proposal
validation. Imported-project state is then merged and revalidated immediately before
the first generation node is dispatched.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .geckolib_generation_contract import (
    GeckoLibGenerationContractError,
    geckolib_entity_inputs_from_module_config,
    preflight_geckolib_generation_target,
)
from .scale_policy import ScalePolicy
from .system_pack_validation import validate_system_modules

_ENTITY_KINDS = frozenset({"entity", "boss", "npc"})
_SYSTEM_PACK_BY_KIND = {
    "quest": "quest-system",
    "class": "class-skill-system",
    "skill": "class-skill-system",
    "economy": "economy-shop",
    "shop": "economy-shop",
    "gui": "gui-networking",
    "networking": "gui-networking",
    "party": "party-guild",
    "guild": "party-guild",
}


class ProductionGenerationPreflightError(ValueError):
    """An approved module set cannot enter deterministic built-in generation."""


def _is_custom(module: Any) -> bool:
    config = getattr(module, "config", None)
    return getattr(module, "kind", None) == "custom_java" or (
        isinstance(config, dict) and config.get("implementation") == "custom"
    )


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
        if _is_custom(module):
            continue
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


def validate_production_generation_modules(
    modules: Iterable[Any],
    *,
    policy: ScalePolicy | None = None,
    validate_system_packs: bool = True,
) -> None:
    """Reject every proposal-known built-in generator failure before dispatch.

    System-pack cross references are proposal-complete only for fresh projects. For an
    imported project, the caller disables that portion here and the project-aware
    preflight merges the persisted system records before validating them.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    materialized = tuple(modules)

    for module in materialized:
        if _is_custom(module):
            continue
        kind = str(module.kind)
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
    for pack_id, group in sorted(_system_groups(materialized).items()):
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
) -> None:
    """Validate project-dependent generator state before concurrent dispatch begins."""

    policy = policy or ScalePolicy.from_environment()
    materialized = tuple(modules)
    validate_production_generation_modules(
        materialized,
        policy=policy,
        validate_system_packs=False,
    )

    if any(
        not _is_custom(module) and str(module.kind) in _ENTITY_KINDS
        for module in materialized
    ):
        try:
            preflight_geckolib_generation_target(
                project_root,
                mod_id=mod_id,
                package_name=package_name,
            )
        except GeckoLibGenerationContractError as exc:
            raise ProductionGenerationPreflightError(
                f"GeckoLib project cannot enter entity generation: {exc}"
            ) from exc

    if not _system_groups(materialized):
        return
    from .system_pack_generator import iter_system_module_records

    for pack_id, group in sorted(_system_groups(materialized).items()):
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
        merged = {
            str(item["module_id"]): item
            for item in existing
        }
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
