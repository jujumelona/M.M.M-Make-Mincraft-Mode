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


def test_published_generator_migration_accepts_only_unreviewed_historical_binding():
    """Source changes migrate only the published, unevidenced generator identity."""
    from minecraft_mod_ai.host_generator_catalog_migration import (
        _OLD_GENERATOR,
        _REVIEWED_GENERATOR,
        _HISTORIC_GENERATORS,
        _rebind_unreviewed_generator,
    )

    assert _HISTORIC_GENERATORS == {_OLD_GENERATOR, _REVIEWED_GENERATOR}
    authority = bootstrap_integrity()
    live = authority.implementations.get_implementation(
        "python_generator:minecraft/screen/registration"
    )
    assert live.content_sha256.startswith("sha256:")
    assert len(live.content_sha256) == len(_REVIEWED_GENERATOR)

    row = {
        "state": "not_reviewed",
        "implementation": {
            "implementation_id": "python_generator:minecraft/screen/registration",
            "implementation_sha256": _REVIEWED_GENERATOR,
            "evidence_id": "",
        },
    }
    changed = _rebind_unreviewed_generator(
        leaf="minecraft/screen/registration",
        row=row,
        actual=live.content_sha256,
        minecraft="26.2",
    )
    assert changed == (live.content_sha256 != _REVIEWED_GENERATOR)
    assert row["implementation"]["implementation_sha256"] == live.content_sha256

    # An admitted/evidenced implementation cannot silently inherit a new hash.
    row["state"] = "admitted"
    row["implementation"]["implementation_sha256"] = _REVIEWED_GENERATOR
    with pytest.raises(ValueError, match="HOST_GENERATOR_BINDING_UNRECOGNIZED"):
        _rebind_unreviewed_generator(
            leaf="minecraft/screen/registration",
            row=row,
            actual="sha256:" + "0" * 64,
            minecraft="26.2",
        )


def test_fabric_26_2_target_selectable_at_session_entry():
    """Exercise the exact version-only and version+loader API preflight paths."""
    from minecraft_mod_ai.api import _validate_requested_target

    assert _validate_requested_target("26.2", None) == ("26.2", None)
    assert _validate_requested_target("26.2", "fabric") == ("26.2", "fabric")


def test_corrupted_host_catalog_is_not_misreported_as_missing_provider(monkeypatch):
    from minecraft_mod_ai import platform_catalog
    from minecraft_mod_ai.api import _validate_requested_target
    from minecraft_mod_ai.resolved_version_context import VersionContextError

    def broken_catalog(_version, _loader):
        raise VersionContextError(
            "HOST_BUNDLE_CATALOG_INVALID",
            reason="HOST_GENERATOR_MIGRATION_REVISION_UNREVIEWED",
        )

    monkeypatch.setattr(platform_catalog, "adapter_for_target", broken_catalog)
    with pytest.raises(VersionContextError, match="HOST_GENERATOR_MIGRATION_REVISION_UNREVIEWED"):
        _validate_requested_target("26.2", None)
