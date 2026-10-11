"""Pinned HOST migration for stale, unreviewed canonical-generator identities.

Only the historically published Python generator fingerprint is eligible.
No admitted evidence, template, validator or schema is rebound.  The new
fingerprint must match the registered local executor exactly.
"""
from __future__ import annotations

from copy import deepcopy

from .resolved_version_context import ResolvedVersionContext, _encode

_OLD_GENERATOR = "sha256:c16768f8ee7c094762ea102bcc8d9b95a85fca854335a2e9209906815d0fa8ab"
_REVIEWED_GENERATOR = "sha256:577f6b964b3096e79afb6537f54134af98391e92efae2d0cfba5159e132d493b"


def migrate_published_generator_authority(catalog: dict) -> dict:
    """Rebind only the pinned stale *not_reviewed* Python candidate identity."""
    from .integrity_bootstrap import get_integrity_authority

    # An unrelated caller-supplied HOST catalog is not a published migration
    # target. In particular never rewrite admitted/custom authority snapshots.
    if len(catalog.get("bundles", ())) != 43:
        return catalog
    authority = get_integrity_authority()
    actual = authority.implementations.get_implementation(
        "python_generator:minecraft/screen/registration"
    ).content_sha256
    if actual != _REVIEWED_GENERATOR:
        raise ValueError("HOST_GENERATOR_MIGRATION_REVISION_UNREVIEWED")

    result = deepcopy(catalog)
    prior_auto = result["auto_context_id"]
    remap = {}
    changed = 0
    for bundle in result["bundles"]:
        bindings = bundle["host_facts"]["leaf_bindings"]
        for leaf, row in bindings.items():
            impl = row.get("implementation")
            if not isinstance(impl, dict):
                continue
            if impl.get("implementation_id") != "python_generator:" + leaf:
                continue
            current = impl.get("implementation_sha256")
            if current == _REVIEWED_GENERATOR:
                continue
            if (
                current != _OLD_GENERATOR
                or row.get("state") != "not_reviewed"
                or impl.get("evidence_id")
            ):
                raise ValueError(
                    "HOST_GENERATOR_BINDING_UNRECOGNIZED: "
                    f"{bundle['target']['minecraft_version']}:{leaf}"
                )
            impl["implementation_sha256"] = _REVIEWED_GENERATOR
            changed += 1
        previous = bundle["context_id"]
        if changed:
            snapshot = {
                "target": bundle["target"],
                "host_facts": bundle["host_facts"],
                "source": bundle["source"],
            }
            new_context = ResolvedVersionContext(_encode(snapshot))
            bundle["context_id"] = new_context.context_id
        remap[previous] = bundle["context_id"]

    if changed == 0:
        return catalog
    if changed != 13449 or prior_auto not in remap:
        raise ValueError("HOST_GENERATOR_MIGRATION_SCOPE_UNRECOGNIZED")
    result["auto_context_id"] = remap[prior_auto]
    return result
