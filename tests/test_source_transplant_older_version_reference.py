"""Older mod source remains inspectable without granting target-code authority."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import source_transplant as source
from minecraft_mod_ai.artifact_dependency_graph import ArtifactKind, ArtifactNode
from minecraft_mod_ai.host_version_catalog import host_target


@pytest.mark.parametrize("unrelated_target_module", (False, True))
def test_real_donor_inspector_returns_version_bound_reference_for_older_source(
    monkeypatch, unrelated_target_module,
):
    java_path = "src/main/java/example/Travel.java"
    java_code = b"package example; public class Travel { public void travel() {} }"
    modern_path = "modern/src/main/java/example/Unrelated.java"
    blobs = {java_path: "b" * 40}
    metadata = {
        "gradle.properties": "minecraft_version=1.20.1",
        "src/main/resources/fabric.mod.json":
            '{"schemaVersion":1,"id":"demo","depends":{"fabricloader":">=0.15.0","minecraft":"1.20.1"}}',
    }
    if unrelated_target_module:
        blobs[modern_path] = "c" * 40
        metadata["modern/gradle.properties"] = "minecraft_version=26.1.2"
        metadata["modern/src/main/resources/fabric.mod.json"] = (
            '{"id":"modern","depends":{"fabricloader":">=0.17.0","minecraft":"26.1.2"}}'
        )

    monkeypatch.setattr(source, "_repository_snapshot", lambda *_args, **_kwargs: {
        "license_id": "MIT",
        "commit_sha": "a" * 40,
        "blobs": blobs,
        "selected_branch": "main",
        "default_branch": "main",
    })
    monkeypatch.setattr(source, "_build_metadata_sources",
                        lambda *_args, **_kwargs: metadata)

    class FakeGraph:
        nodes = {path: ArtifactNode(path, ArtifactKind.JAVA_SOURCE) for path in blobs}
        unresolved_edges = ()
        ambiguous_edges = ()
        adjacency = {}

        def compute_directional_closures(self, seeds):
            return (tuple(seeds),)

        def is_closure_complete(self, selected):
            return tuple(selected) == (java_path,)

    class FakeIndex:
        metadata = {}
        def build_dependency_graph(self):
            return FakeGraph()
        def artifact_bytes(self, path):
            assert path == java_path
            return java_code

    monkeypatch.setattr(source, "_github_client",
                        lambda *_args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(source.RepositoryArtifactIndex, "build_from_tree",
                        lambda *_args, **_kwargs: FakeIndex())
    monkeypatch.setattr(source.CapabilityImplementationLocator, "locate_seeds",
                        lambda *_args: (SimpleNamespace(
                            node_id=java_path, evidence_types=("source_symbol",)),))
    adapter = host_target("26.1.2")
    donor = source.inspect_repository_slice(
        repository="example/travel", capability="space travel", adapter=adapter,
        discovery_client=SimpleNamespace(github_token=""),
    )
    assert donor is not None
    assert donor.compatibility_evidence.minecraft_version == "1.20.1"
    assert donor.compatibility_evidence.loader == "fabric"
    assert donor.target_compatibility == "reference_only"
    source.validate_donor_slice_manifest(donor)
    assert donor.files[0].sha256.startswith("sha256:")

    # Strict executable reuse remains denied even if a forged receipt were
    # supplied for this old-version candidate.
    from minecraft_mod_ai.grounded_source_reuse import _proof_admits
    from minecraft_mod_ai.reuse_proof_executor import ReuseProofReceipt
    receipt = ReuseProofReceipt(
        candidate_id=f"{donor.repository}@{donor.commit_sha}",
        capability=donor.capability, commit_sha=donor.commit_sha,
        closure_hash=source.donor_closure_sha256(donor), proof_level="COMPILE_VERIFIED",
        compile_passed=True, tests_passed=True, authoritative_compile=True,
        verified_capabilities=(donor.capability,), verified_artifacts=(java_path,),
        unresolved_symbols=(), missing_resources=(), adaptations_applied=(),
        residual_capabilities=(),
    )
    assert _proof_admits(donor, receipt) is False
