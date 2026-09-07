from __future__ import annotations

"""Pure proposal-level preflight for deterministic built-in generation inputs.

The complete orchestrator may execute independent generation nodes concurrently. Any
configuration error that can be determined from the approved proposal must therefore
be rejected before the first node is dispatched, not after another node has already
written files.
"""

from collections.abc import Iterable
from typing import Any

from .geckolib_generation_contract import (
    GeckoLibGenerationContractError,
    geckolib_entity_inputs_from_module_config,
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


def validate_production_generation_modules(
    modules: Iterable[Any],
    *,
    policy: ScalePolicy | None = None,
) -> None:
    """Reject every proposal-known built-in generator failure before dispatch."""

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    system_groups: dict[str, list[Any]] = {}

    for module in modules:
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
        pack_id = _SYSTEM_PACK_BY_KIND.get(kind)
        if pack_id is not None:
            system_groups.setdefault(pack_id, []).append(module)

    for pack_id in sorted(system_groups):
        try:
            validate_system_modules(
                pack_id,
                [_system_module_dict(module) for module in system_groups[pack_id]],
            )
        except (TypeError, ValueError) as exc:
            raise ProductionGenerationPreflightError(
                f"System pack {pack_id} cannot enter generation: {exc}"
            ) from exc


__all__ = [
    "ProductionGenerationPreflightError",
    "validate_production_generation_modules",
]
