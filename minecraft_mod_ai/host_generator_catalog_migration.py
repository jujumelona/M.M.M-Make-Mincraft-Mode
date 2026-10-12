"""Rebind unreviewed HOST Python generator candidates to the installed executor.

The packaged HOST catalog contains immutable historical implementation identities.
Unlike admitted implementations, not_reviewed Python generators have no executable
approval/evidence to preserve. Their current registration must follow the actual
source fingerprint of the installed, integrity-owned generator after a code change.

Only the known published 43-bundle catalog and its two historical source identities
are eligible. No admitted, evidenced, unsupported, foreign or unknown binding is
rewritten. Immutable context IDs are recomputed from their actual contents.
"""
from __future__ import annotations

from copy import deepcopy

from .resolved_version_context import ResolvedVersionContext, _encode

_OLD_GENERATOR = "sha256:c16768f8ee7c094762ea102bcc8d9b95a85fca854335a2e9209906815d0fa8ab"
_REVIEWED_GENERATOR = "sha256:544d6b43e511f3b37be8b28cfa80c849f97cf06340b98c92eab84221197f0576"
_HISTORIC_GENERATORS = frozenset({_OLD_GENERATOR, _REVIEWED_GENERATOR})
_PUBLISHED_BUNDLES = 43
_PUBLISHED_GENERATOR_BINDINGS = 13449


def _rebind_unreviewed_generator(
    *,
    leaf: str,
    row: dict,
    actual: str,
    minecraft: str,
) -> bool:
    """Update one known, unevidenced Python candidate; never reapprove anything."""
    impl = row.get("implementation")
    if not isinstance(impl, dict):
        return False
    if impl.get("implementation_id") != "python_generator:" + leaf:
        return False

    current = impl.get("implementation_sha256")
    if current == actual:
        return False
    if (
        current not in _HISTORIC_GENERATORS
        or row.get("state") != "not_reviewed"
        or impl.get("evidence_id")
    ):
        raise ValueError(
            "HOST_GENERATOR_BINDING_UNRECOGNIZED: "
            f"{minecraft}:{leaf}"
        )
    impl["implementation_sha256"] = actual
    return True


def migrate_published_generator_authority(catalog: dict) -> dict:
    """Upgrade only preexisting, not_reviewed candidate identities to current code."""
    # This migration is not a general license to change HOST version facts.
    bundles = catalog.get("bundles", ())
    if not isinstance(bundles, list) or len(bundles) != _PUBLISHED_BUNDLES:
        return catalog

    from .integrity_bootstrap import get_integrity_authority

    authority = get_integrity_authority()
    # The authority's registered executor is hashed from the actual installed
    # function source, not a migration constant that breaks on every code edit.
    implementation = authority.implementations.get_implementation(
        "python_generator:minecraft/screen/registration"
    )
    actual = implementation.content_sha256
    if implementation.executor_type.value != "python_generator":
        raise ValueError("HOST_GENERATOR_EXECUTOR_TYPE_UNRECOGNIZED")

    result = deepcopy(catalog)
    prior_auto = result["auto_context_id"]
    remap: dict[str, str] = {}
    total = 0
    changed = 0
    for bundle in result["bundles"]:
        version = str(bundle["target"]["minecraft_version"])
        bindings = bundle["host_facts"]["leaf_bindings"]
        changed_here = False
        for leaf, row in bindings.items():
            impl = row.get("implementation")
            if not isinstance(impl, dict) or impl.get("implementation_id") != "python_generator:" + leaf:
                continue
            total += 1
            if _rebind_unreviewed_generator(
                leaf=leaf, row=row, actual=actual, minecraft=version,
            ):
                changed += 1
                changed_here = True
        previous = bundle["context_id"]
        if changed_here:
            snapshot = {
                "target": bundle["target"],
                "host_facts": bundle["host_facts"],
                "source": bundle["source"],
            }
            bundle["context_id"] = ResolvedVersionContext(_encode(snapshot)).context_id
        remap[previous] = bundle["context_id"]

    # A 43-entry foreign catalog must not inherit published binding authority.
    # Check even when all generator identities already match this checkout.
    if total != _PUBLISHED_GENERATOR_BINDINGS or prior_auto not in remap:
        raise ValueError(
            "HOST_GENERATOR_MIGRATION_SCOPE_UNRECOGNIZED: "
            f"actual_bindings={total}"
        )
    if changed == 0:
        return catalog
    result["auto_context_id"] = remap[prior_auto]
    return result
