from __future__ import annotations

"""Canonical projection of resolved HOST version facts into template execution context."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .resolved_version_context import ResolvedVersionContext, VersionContextError
from .target_contract import target_contract_from_mapping


# Every deterministic HOST domain exposed by ResolvedVersionContext must cross the
# template boundary unchanged. Templates may consume these values, but callers/models
# may not re-derive, default, or override them locally.
_HOST_FACT_KEYS = (
    "host_revision",
    "capabilities",
    "api_symbols",
    "schemas",
    "artifact_rules",
    "dependency_coordinates",
    "repositories",
    "replacements",
    "leaf_bindings",
)


def resolved_template_facts(resolved: ResolvedVersionContext) -> dict[str, Any]:
    """Return every immutable fact templates may consume from one resolved HOST snapshot.

    Target semantics are normalized only by ``TargetContract``. This projection layer is
    deliberately policy-free: it does not parse Minecraft versions or infer mappings,
    naming, Java, pack, or provider coordinates itself.
    """
    payload = resolved.to_dict()
    target = dict(payload["target"])
    host_facts = payload.get("host_facts") or {}
    if not isinstance(host_facts, Mapping):
        raise VersionContextError(
            "HOST_FACTS_INVALID",
            actual=type(host_facts).__name__,
            context_id=resolved.context_id,
        )

    missing = [key for key in _HOST_FACT_KEYS if key not in host_facts]
    if missing:
        raise VersionContextError(
            "HOST_BUNDLE_INCOMPLETE",
            fields=missing,
            context_id=resolved.context_id,
        )

    # Rehydrate through the canonical target authority instead of reconstructing
    # deterministic version rules at the template boundary.
    public_target = target_contract_from_mapping(target).public_dict()
    public_target.pop("host_facts_json", None)
    naming = public_target.pop("naming_regime")
    pack_versions = public_target.pop("pack_versions")

    canonical: dict[str, Any] = {
        **public_target,
        "version_context_id": resolved.context_id,
        "mappings_applicable": naming["mappings_applicable"],
        "naming_regime": naming["kind"],
        "pack_versions": pack_versions,
    }

    for key in _HOST_FACT_KEYS:
        canonical[key] = deepcopy(host_facts[key])
    return canonical


def validate_resolved_template_overrides(
    resolved: ResolvedVersionContext,
    *sources: Mapping[str, Any],
) -> None:
    """Reject caller/model attempts to contradict any projected immutable HOST fact."""
    canonical = resolved_template_facts(resolved)
    for source in sources:
        for key, expected in canonical.items():
            if key in source and source[key] != expected:
                raise VersionContextError(
                    "HOST_FACT_OVERRIDE",
                    field=key,
                    expected=expected,
                    actual=source[key],
                    context_id=resolved.context_id,
                )


def resolved_template_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return execution values with immutable HOST facts prefilled.

    The resolved HOST snapshot is the sole authority for target coordinates and every
    deterministic capability/API/schema/artifact/dependency/leaf binding fact. Callers
    may omit those facts, but may not contradict them. Project-specific semantic values
    remain untouched.
    """
    merged = dict(values)
    raw = merged.get("resolved_version_context")
    if raw is None:
        return merged

    resolved = raw if isinstance(raw, ResolvedVersionContext) else ResolvedVersionContext.from_dict(raw)
    canonical = resolved_template_facts(resolved)
    validate_resolved_template_overrides(resolved, merged)
    for key, expected in canonical.items():
        if key not in merged:
            merged[key] = deepcopy(expected)
    return merged


__all__ = [
    "resolved_template_facts",
    "resolved_template_values",
    "validate_resolved_template_overrides",
]
