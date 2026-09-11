"""Offline checks for official metadata; no fabricated Minecraft runtime passes."""

from hashlib import sha256
import json

import pytest

from minecraft_mod_ai.host_version_catalog import DEFAULT_CATALOG, host_target, load_host_catalog
from minecraft_mod_ai.resolved_version_context import VersionContextError, VersionRequest


def test_packaged_catalog_has_exact_requests_and_evidence_bindings(monkeypatch):
    monkeypatch.delenv("MMM_VERSION_BUNDLE_CATALOG", raising=False)
    resolver, bundles = load_host_catalog()
    report = json.loads(DEFAULT_CATALOG.with_name("official_version_evidence.json").read_text())
    evidence = {row["context_id"]: row for row in report["versions"]}
    assert len(bundles) == len(evidence) == 43
    assert resolver.resolve(VersionRequest()).minecraft == "26.2"
    for context in bundles:
        assert resolver.resolve(VersionRequest.from_input(context.minecraft)) == context
        assert host_target(context.minecraft).version_context == context
        row = dict(evidence[context.context_id])
        row.pop("context_id")
        digest = sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert context.host_revision == "sha256:" + digest
        assert row["compile"] == row["gametest"] == "not_run"
        assert row["minecraft_java_minimum"] == context.java
        assert int(context.facts["dependency_coordinates"]["gradle_jvm_major"]) >= max(context.java, 21)
        assert row["loom_selection"]["fixed_release"] == context.target["fabric_loom"]
        assert row["loom_selection"]["policy"] == "fixed_release_binary_and_dependencies_match_official_recommendation"
        assert len(row["loom_selection"]["binary_sha256"]) == 64
        assert not context.facts["artifact_rules"]
    assert all(len(source["sha256"]) == 64 and source["bytes"] > 0 for source in report["sources"].values())
    assert {row["minecraft"] for row in report["skipped"]} == {"1.14", "1.14.1", "1.14.2", "1.14.3"}
    for row in report["skipped"]:
        partial = row["verified_partial_facts"]
        assert partial["minecraft_java_minimum"] == 8
        # Early Mojang embedded IDs include a build UUID; release_target is exact.
        assert partial["mojang_version_json"]["release_target"] == row["minecraft"]
        assert not partial["mojang_mappings_available"]
        assert partial["published_yarn_coordinates"]


@pytest.mark.parametrize("version,java,data,resource", [
    ("1.14.4", 8, "4", "4"), ("1.20.1", 17, "15", "15"),
    ("1.21.1", 21, "48", "34"), ("1.21.4", 21, "61", "46"),
    ("26.1", 25, "101.1", "84.0"), ("26.2", 25, "107.1", "88.0"),
])
def test_official_pack_and_java_coordinates(monkeypatch, version, java, data, resource):
    monkeypatch.delenv("MMM_VERSION_BUNDLE_CATALOG", raising=False)
    target = host_target(version)
    assert (target.java_version, target.data_pack_version, target.resource_pack_version) == (str(java), data, resource)
    assert target.mappings_applicable == version.startswith("1.")


def test_unreviewed_capability_is_not_authorized_by_metadata(monkeypatch):
    monkeypatch.delenv("MMM_VERSION_BUNDLE_CATALOG", raising=False)
    context = host_target("auto").version_context
    with pytest.raises(VersionContextError):
        context.require_capability("REGISTER_ITEM")
    with pytest.raises(VersionContextError, match="UNSUPPORTED_MINECRAFT_VERSION"):
        host_target("1.14.3")
