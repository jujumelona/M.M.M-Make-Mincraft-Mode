from __future__ import annotations

"""Canonical projection of resolved HOST version facts into template/model context."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .resolved_version_context import ResolvedVersionContext, VersionContextError
from .target_contract import mappings_applicable


def resolved_template_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return execution values with immutable version-coupled facts prefilled.

    The resolved HOST snapshot is the sole authority for target/build coordinates.  Callers
    may omit those facts, but may not supply a contradictory value.  Project-specific
    semantic values remain untouched.
    """
    merged = dict(values)
    raw = merged.get("resolved_version_context")
    if raw is None:
        return merged

    resolved = raw if isinstance(raw, ResolvedVersionContext) else ResolvedVersionContext.from_dict(raw)
    target = dict(resolved.to_dict()["target"])
    mapping_is_applicable = mappings_applicable(target["minecraft_version"])
    canonical = {
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

    for key, expected in canonical.items():
        if key in merged and merged[key] != expected:
            raise VersionContextError(
                "HOST_FACT_OVERRIDE",
                field=key,
                expected=expected,
                actual=merged[key],
                context_id=resolved.context_id,
            )
        if key not in merged:
            merged[key] = deepcopy(expected)
    return merged


__all__ = ["resolved_template_values"]
