from __future__ import annotations

"""Canonical projection of resolved HOST version facts into template execution context."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .resolved_version_context import ResolvedVersionContext, VersionContextError
from .target_contract import mappings_applicable


# These catalog sections are immutable, version-coupled build facts.  They belong in
# deterministic template execution, not in model-authored project semantics.
_ECOSYSTEM_FACT_KEYS = (
    "dependency_coordinates",
    "repositories",
    "replacements",
)


def resolved_template_facts(resolved: ResolvedVersionContext) -> dict[str, Any]:
    """Return every immutable fact templates may consume from one resolved HOST snapshot."""
    payload = resolved.to_dict()
    target = dict(payload["target"])
    host_facts = payload.get("host_facts") or {}
    if not isinstance(host_facts, Mapping):
        raise VersionContextError(
            "HOST_FACTS_INVALID",
            actual=type(host_facts).__name__,
            context_id=resolved.context_id,
        )

    mapping_is_applicable = mappings_applicable(target["minecraft_version"])
    canonical: dict[str, Any] = {
        **target,
        "version_context_id": resolved.context_id,
        "mappings_applicable": mapping_is_applicable,
        "naming_regime": "mapped_obfuscated" if mapping_is_applicable else "native_unobfuscated",
        "pack_versions": {
            "data": target["data_pack_version"],
            "resource": target["resource_pack_version"],
            "resource_major": target["resource_pack_format"],
        },
    }
    if mapping_is_applicable:
        canonical["mappings"] = {
            "kind": target["mappings_kind"],
            "version": target["mappings_version"],
        }

    # Keep large catalogs such as assets/templates/invariants HOST-side.  Only compact
    # build facts that deterministic templates need are projected into execution values.
    for key in _ECOSYSTEM_FACT_KEYS:
        if key in host_facts:
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
    """Return execution values with immutable version/build facts prefilled.

    The resolved HOST snapshot is the sole authority for target coordinates and compact
    ecosystem/build facts. Callers may omit those facts, but may not contradict them.
    Project-specific semantic values remain untouched.
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
