"""Compare actual published HOST leaf identities with current executor registrations.

This runs without Minecraft or model downloads.  A CI pass must mean the
published HOST contract can reach runtime verification unchanged.
"""
from __future__ import annotations

import pytest

from minecraft_mod_ai.host_version_catalog import host_target
from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity


@pytest.mark.parametrize("version", ["1.21.1", "1.21.4", "26.2"])
def test_published_leaf_hashes_match_live_registry(version):
    authority = bootstrap_integrity()
    resolved = host_target(version).version_context
    mismatches = []
    for leaf, binding in sorted(resolved.leaf_bindings.items()):
        if binding["state"] == "unsupported":
            continue
        impl = binding["implementation"]
        live = authority.implementations.get_implementation(impl["implementation_id"])
        expected = {
            "implementation_sha256": live.content_sha256,
            "validator_sha256": authority.implementations.get_validator(
                impl["validator_profile"]
            ).source_hash,
            "input_schema_sha256": authority.types.get_schema_hash(leaf + ":input"),
            "output_schema_sha256": authority.types.get_schema_hash(leaf + ":output"),
        }
        for field, current in expected.items():
            if impl.get(field) != current:
                mismatches.append(f"{leaf}:{field}")
    assert not mismatches, (
        f"RUNTIME_INTEGRITY_BINDING_MISMATCH version={version} "
        f"fields={mismatches[:40]} total={len(mismatches)}"
    )


def test_live_runtime_candidate_matches_catalog_for_modular_gui():
    from minecraft_mod_ai.artifact_job import ArtifactJob
    from minecraft_mod_ai.integrity_dispatcher import verify_job_binding

    resolved = host_target("26.2").version_context
    leaf = "minecraft/screen/registration"
    row = resolved.leaf_bindings[leaf]
    assert row["state"] == "not_reviewed"
    impl = row["implementation"]
    job = ArtifactJob(
        "launch_readiness_screen_gui.minecraft_screen_registration",
        "", "launch_readiness_screen_gui", canonical_leaf=leaf,
        implementation_id=impl["implementation_id"],
    )
    assert verify_job_binding(job, resolved, {}) is not None
