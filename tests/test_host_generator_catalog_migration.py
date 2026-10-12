"""The packaged HOST generator migration must survive routine source edits.

No test executes a model, rewrites admitted authority, or creates evidence.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import host_generator_catalog_migration as migration


NEW_SOURCE = "sha256:" + "a" * 64


def _row(old_sha, state="not_reviewed", evidence=None):
    return {
        "state": state,
        "implementation": {
            "implementation_id": "python_generator:minecraft/screen/registration",
            "executor_type": "generator_handoff_screen",
            "implementation_sha256": old_sha,
            "evidence_id": evidence,
        },
    }


def _catalog(row):
    bundles = []
    for n in range(43):
        bindings = {"minecraft/screen/registration": deepcopy(row)} if n == 0 else {}
        bundles.append({
            "context_id": f"before-{n}",
            "target": {"minecraft_version": str(n)},
            "host_facts": {"leaf_bindings": bindings},
            "source": "HOST",
        })
    return {
        "schema_version": "mmm/host-version-catalog-v1",
        "bundles": bundles,
        "auto_context_id": "before-0",
    }


def _install_fake_runtime(monkeypatch):
    import minecraft_mod_ai.integrity_bootstrap as bootstrap

    class FakeImplementationRegistry:
        def get_implementation(self, name):
            assert name == "python_generator:minecraft/screen/registration"
            return SimpleNamespace(
                content_sha256=NEW_SOURCE,
                executor_type=SimpleNamespace(value="python_generator"),
            )

    monkeypatch.setattr(
        bootstrap,
        "get_integrity_authority",
        lambda: SimpleNamespace(implementations=FakeImplementationRegistry()),
    )
    monkeypatch.setattr(migration, "_PUBLISHED_GENERATOR_BINDINGS", 1)

    class FakeResolvedContext:
        def __init__(self, snapshot_json):
            self.context_id = "sha256:" + sha256(snapshot_json.encode()).hexdigest()

    monkeypatch.setattr(migration, "ResolvedVersionContext", FakeResolvedContext)


@pytest.mark.parametrize(
    "historical_sha",
    [migration._OLD_GENERATOR, migration._REVIEWED_GENERATOR],
)
def test_unevidenced_historical_candidate_tracks_installed_generator(monkeypatch, historical_sha):
    _install_fake_runtime(monkeypatch)
    original = _catalog(_row(historical_sha))
    saved = deepcopy(original)
    result = migration.migrate_published_generator_authority(original)
    binding = result["bundles"][0]["host_facts"]["leaf_bindings"]["minecraft/screen/registration"]
    assert binding["implementation"]["implementation_sha256"] == NEW_SOURCE
    assert binding["state"] == "not_reviewed"
    assert binding["implementation"]["evidence_id"] is None
    assert result["auto_context_id"] == result["bundles"][0]["context_id"]
    assert result["auto_context_id"] != "before-0"
    assert result["bundles"][1]["context_id"] == "before-1"
    assert original == saved


def test_current_generator_catalog_is_idempotent(monkeypatch):
    _install_fake_runtime(monkeypatch)
    catalog = _catalog(_row(NEW_SOURCE))
    assert migration.migrate_published_generator_authority(catalog) is catalog


@pytest.mark.parametrize("state,evidence", [
    ("admitted", None),
    ("not_reviewed", "proof-123"),
    ("unsupported", None),
])
def test_never_rewrite_admitted_evidenced_or_unsupported(monkeypatch, state, evidence):
    _install_fake_runtime(monkeypatch)
    catalog = _catalog(_row(migration._OLD_GENERATOR, state=state, evidence=evidence))
    before = deepcopy(catalog)
    with pytest.raises(ValueError, match="HOST_GENERATOR_BINDING_UNRECOGNIZED"):
        migration.migrate_published_generator_authority(catalog)
    assert catalog == before


def test_unknown_historical_sha_is_rejected(monkeypatch):
    _install_fake_runtime(monkeypatch)
    with pytest.raises(ValueError, match="HOST_GENERATOR_BINDING_UNRECOGNIZED"):
        migration.migrate_published_generator_authority(_catalog(_row("sha256:" + "0" * 64)))


def test_foreign_43_bundle_catalog_cannot_inherit_published_generator_scope(monkeypatch):
    _install_fake_runtime(monkeypatch)
    monkeypatch.setattr(migration, "_PUBLISHED_GENERATOR_BINDINGS", 2)
    with pytest.raises(ValueError, match="HOST_GENERATOR_MIGRATION_SCOPE_UNRECOGNIZED"):
        migration.migrate_published_generator_authority(
            _catalog(_row(migration._OLD_GENERATOR))
        )
