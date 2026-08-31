from __future__ import annotations

import hashlib

import pytest

import minecraft_mod_ai.reuse_proof_executor as reuse_proof


@pytest.fixture(autouse=True)
def _worker6_isolate_scaffold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reuse_proof,
        "scaffold_minimal_ephemeral_workspace",
        lambda *args, **kwargs: None,
    )
from minecraft_mod_ai import source_transplant
from minecraft_mod_ai.proof_level import ProofLevel, validate_proof_transition
from minecraft_mod_ai.reuse_artifacts import (
    ReusableArtifactBundle,
    bundle_proof_allows_reuse,
)
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice


def _donor(capability: str = "boss.entity") -> DonorSlice:
    payload = b"package donor; public class BossEntity {}\n"
    return DonorSlice(capability=capability, repository="example/authority", commit_sha="a" * 40, license_id="MIT", source_url="https://github.com/example/authority", target_compatibility="metadata_exact", files=(DonorFile(path="src/main/java/donor/BossEntity.java", blob_sha="b" * 40, sha256="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload), symbols=("BossEntity",)),), seed_files=("src/main/java/donor/BossEntity.java",), source_symbols=("BossEntity",), required_dependencies=(), donor_tests=(), confidence=0.9, closure_complete=True)


def test_compile_transition_requires_authoritative_executor():
    valid, reason = validate_proof_transition(ProofLevel.MATERIALIZED, ProofLevel.COMPILE_VERIFIED, receipt={"compile_passed": True})
    assert valid is False and "authoritative" in reason
    valid, _ = validate_proof_transition(ProofLevel.MATERIALIZED, ProofLevel.COMPILE_VERIFIED, receipt={"compile_passed": True, "authoritative_compile": True})
    assert valid is True


def test_subgraph_transition_requires_authoritative_executor():
    valid, reason = validate_proof_transition(ProofLevel.MATERIALIZED, ProofLevel.SUBGRAPH_COMPILE_VERIFIED, receipt={"verified_subgraphs": 1})
    assert valid is False and "authoritative" in reason
    valid, _ = validate_proof_transition(ProofLevel.MATERIALIZED, ProofLevel.SUBGRAPH_COMPILE_VERIFIED, receipt={"verified_subgraphs": 1, "authoritative_compile": True})
    assert valid is True


def test_compile_checker_is_diagnostic_only(monkeypatch, tmp_path):
    payload = b"package donor; public class BossEntity {}\n"
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", lambda *args, **kwargs: {"src/main/java/donor/BossEntity.java": payload})
    receipt = reuse_proof.execute_reuse_proof(_donor(), target_workspace=tmp_path, target_context={}, compile_checker=lambda *_: {"compile_passed": True, "tests_passed": True})
    assert receipt.authoritative_compile is False
    assert receipt.compile_passed is False
    assert not ProofLevel.from_value(receipt.proof_level).allows_reuse()
    assert receipt.failure_scope == "verification"
    assert receipt.failure_code == "NON_AUTHORITATIVE_COMPILE_CHECKER"


def test_fallback_rejects_capability_mismatch_without_proof_execution(monkeypatch, tmp_path):
    donor = _donor("other.capability")
    monkeypatch.setattr(reuse_proof, "execute_reuse_proof", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute")))
    selected, receipts = reuse_proof.execute_candidate_fallback_loop((donor,), "boss.entity", target_workspace=tmp_path, target_context={})
    assert selected is None
    assert len(receipts) == 1
    assert receipts[0].failure_scope == "donor"
    assert receipts[0].failure_code == "CAPABILITY_MISMATCH"


def test_fallback_requires_nonempty_capability(tmp_path):
    with pytest.raises(ValueError, match="capability must be non-empty"):
        reuse_proof.execute_candidate_fallback_loop((), "  ", target_workspace=tmp_path, target_context={})


def test_external_bundle_rejects_non_authoritative_compile_receipt():
    donor = _donor()
    path, digest = donor.files[0].path, donor.files[0].sha256
    source_ref = f"{donor.repository}@{donor.commit_sha}"
    bundle = ReusableArtifactBundle(bundle_id="donor:test", capability=donor.capability, origin_kind="external_donor", source_ref=source_ref, file_hashes={path: digest}, protected_paths=(path,), provenance={"repository": donor.repository, "commit_sha": donor.commit_sha, "license_id": donor.license_id, "donor_slice": donor.to_dict()})
    base_receipt = {"proof_level": ProofLevel.COMPILE_VERIFIED.value, "capability": donor.capability, "candidate_id": source_ref, "compile_passed": True, "dependency_receipts": (), "contract": {"protected_artifacts": {path: digest}}}
    assert bundle_proof_allows_reuse(bundle, base_receipt) is False
    assert bundle_proof_allows_reuse(bundle, dict(base_receipt, authoritative_compile=True)) is True


def test_invalid_manifest_has_structured_donor_failure(monkeypatch, tmp_path):
    donor = _donor()
    invalid = DonorSlice(**{**donor.__dict__, "commit_sha": "not-a-pin"})
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not materialize")))
    receipt = reuse_proof.execute_reuse_proof(invalid, target_workspace=tmp_path, target_context={}, compile_checker=lambda *_: True)
    assert receipt.failure_scope == "donor"
    assert receipt.failure_code == "MANIFEST_INVALID"
    assert receipt.compile_passed is False
