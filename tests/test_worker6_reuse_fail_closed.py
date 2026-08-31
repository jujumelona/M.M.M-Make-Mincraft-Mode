from __future__ import annotations

import hashlib
from dataclasses import replace

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
from minecraft_mod_ai.source_transplant import (
    DonorFile,
    DonorSlice,
    SourceTransplantError,
)


def _donor(
    *,
    repository: str = "example/worker6-donor",
    license_id: str = "MIT",
    required_dependencies: tuple[str, ...] = (),
) -> DonorSlice:
    payload = b"package donor; public class BossEntity {}\n"
    return DonorSlice(
        capability="boss.entity",
        repository=repository,
        commit_sha="a" * 40,
        license_id=license_id,
        source_url=f"https://github.com/{repository}",
        target_compatibility="metadata_exact",
        files=(
            DonorFile(
                path="src/main/java/donor/BossEntity.java",
                blob_sha="b" * 40,
                sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                symbols=("BossEntity",),
            ),
        ),
        seed_files=("src/main/java/donor/BossEntity.java",),
        source_symbols=("BossEntity",),
        required_dependencies=required_dependencies,
        donor_tests=(),
        confidence=0.95,
        closure_complete=True,
    )


def _receipt(
    candidate: DonorSlice,
    *,
    level: ProofLevel,
    compile_passed: bool,
) -> reuse_proof.ReuseProofReceipt:
    return reuse_proof.ReuseProofReceipt(
        candidate_id=f"{candidate.repository}@{candidate.commit_sha}",
        capability=candidate.capability,
        commit_sha=candidate.commit_sha,
        closure_hash="sha256:test",
        proof_level=level.value,
        compile_passed=compile_passed,
        tests_passed=False,
        unresolved_symbols=(),
        missing_resources=(),
        adaptations_applied=(),
        verified_capabilities=(),
        residual_capabilities=(candidate.capability,),
    )


def _materialized_payload() -> dict[str, bytes]:
    return {
        "src/main/java/donor/BossEntity.java": (
            b"package donor; public class BossEntity {}\n"
        )
    }


def _authoritative_compile_receipt(
    receipt: reuse_proof.ReuseProofReceipt,
    donor: DonorSlice,
) -> reuse_proof.ReuseProofReceipt:
    protected_path = "src/main/java/ai/minecraft/generated/mod/BossEntity.java"
    adapted_payload = b"package ai.minecraft.generated.mod; public class BossEntity {}\n"
    protected_artifacts = {
        protected_path: "sha256:" + hashlib.sha256(adapted_payload).hexdigest()
    }
    return replace(
        receipt,
        proof_level=ProofLevel.COMPILE_VERIFIED.value,
        compile_passed=True,
        authoritative_compile=True,
        verified_capabilities=(donor.capability,),
        residual_capabilities=(),
        verified_artifacts=(protected_path,),
        verified_symbols=("BossEntity",),
        contract={"protected_artifacts": protected_artifacts},
    )


def test_license_verified_requires_discovery_admitted_license() -> None:
    valid, reason = validate_proof_transition(
        ProofLevel.DISCOVERED,
        ProofLevel.LICENSE_VERIFIED,
        receipt={"license": "GPL-3.0"},
    )
    assert valid is False
    assert "reusable source license" in reason

    valid, reason = validate_proof_transition(
        ProofLevel.DISCOVERED,
        ProofLevel.LICENSE_VERIFIED,
        receipt={"license": "MIT"},
    )
    assert valid is True
    assert reason == "transition_valid"


def test_pinned_requires_immutable_full_commit_sha() -> None:
    valid, reason = validate_proof_transition(
        ProofLevel.LICENSE_VERIFIED,
        ProofLevel.PINNED,
        receipt={"commit_sha": "abc123"},
    )
    assert valid is False
    assert "immutable 40-64 hex commit_sha" in reason

    valid, reason = validate_proof_transition(
        ProofLevel.LICENSE_VERIFIED,
        ProofLevel.PINNED,
        receipt={"commit_sha": "a" * 40},
    )
    assert valid is True
    assert reason == "transition_valid"


def test_direct_donor_license_bypass_stops_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    called = False

    def should_not_materialize(*args, **kwargs):
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("invalid-license donor reached materialization")

    monkeypatch.setattr(
        source_transplant,
        "materialize_pinned_donor",
        should_not_materialize,
    )

    receipt = reuse_proof.execute_reuse_proof(
        _donor(license_id="GPL-3.0"),
        target_workspace=tmp_path,
        target_context={},
        compile_checker=lambda *_: True,
    )

    assert called is False
    assert receipt.proof_level == ProofLevel.DISCOVERED.value
    assert receipt.compile_passed is False
    assert receipt.verified_capabilities == ()


def test_fallback_rejects_compile_bool_without_verified_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    unverified = _donor(repository="example/unverified")
    verified = _donor(repository="example/verified")
    results = iter(
        (
            _receipt(
                unverified,
                level=ProofLevel.DISCOVERED,
                compile_passed=True,
            ),
            _receipt(
                verified,
                level=ProofLevel.COMPILE_VERIFIED,
                compile_passed=True,
            ),
        )
    )

    monkeypatch.setattr(
        reuse_proof,
        "execute_reuse_proof",
        lambda *args, **kwargs: next(results),
    )

    selected, receipts = reuse_proof.execute_candidate_fallback_loop(
        (unverified, verified),
        "boss.entity",
        target_workspace=tmp_path,
        target_context={},
    )

    assert selected is verified
    assert len(receipts) == 2
    assert receipts[0].proof_level == ProofLevel.DISCOVERED.value
    assert receipts[1].proof_level == ProofLevel.COMPILE_VERIFIED.value


def test_source_transplant_failure_is_isolated_to_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def donor_failure(*args, **kwargs):
        del args, kwargs
        raise SourceTransplantError("donor blob hash mismatch")

    monkeypatch.setattr(
        source_transplant,
        "materialize_pinned_donor",
        donor_failure,
    )

    receipt = reuse_proof.execute_reuse_proof(
        _donor(),
        target_workspace=tmp_path,
        target_context={},
    )

    assert receipt.compile_passed is False
    assert receipt.verified_capabilities == ()
    assert receipt.residual_capabilities == ("boss.entity",)


def test_unexpected_materialization_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def programming_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("worker6 sentinel programming failure")

    monkeypatch.setattr(
        source_transplant,
        "materialize_pinned_donor",
        programming_failure,
    )

    with pytest.raises(RuntimeError, match="worker6 sentinel programming failure"):
        reuse_proof.execute_reuse_proof(
            _donor(),
            target_workspace=tmp_path,
            target_context={},
        )


def test_dependency_resolution_receipt_replaces_donor_version_and_binds_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    donor = _donor(
        required_dependencies=(
            "software.bernie.geckolib:geckolib-fabric:0.0.1",
        )
    )
    monkeypatch.setattr(
        source_transplant,
        "materialize_pinned_donor",
        lambda *args, **kwargs: _materialized_payload(),
    )

    diagnostic_receipt = reuse_proof.execute_reuse_proof(
        donor,
        target_workspace=tmp_path,
        target_context={
            "loader": "fabric",
            "minecraft_version": "1.21.1",
            "java_version": 21,
            "target_package": "ai.minecraft.generated.mod",
            "target_modid": "generated_mod",
        },
        compile_checker=lambda *_: True,
    )

    assert ProofLevel.from_value(diagnostic_receipt.proof_level).is_verified() is False
    assert diagnostic_receipt.compile_passed is False
    assert diagnostic_receipt.authoritative_compile is False
    assert len(diagnostic_receipt.dependency_receipts) == 1
    dependency = diagnostic_receipt.dependency_receipts[0]
    assert dependency["donor_declared_coordinate"].endswith(":0.0.1")
    assert dependency["resolved_coordinate"] == (
        "software.bernie.geckolib:geckolib-fabric:4.6.0"
    )
    assert dependency["selected_version"] == "4.6.0"
    assert dependency["is_resolved"] is True
    assert dependency["resolution_fingerprint"].startswith("sha256:")

    authoritative_receipt = _authoritative_compile_receipt(diagnostic_receipt, donor)
    protected_artifacts = authoritative_receipt.contract["protected_artifacts"]
    bundle = ReusableArtifactBundle.from_donor_slice(
        donor,
        proof_receipt=authoritative_receipt,
        protected_artifacts=protected_artifacts,
    )
    assert bundle.dependency_receipts == authoritative_receipt.dependency_receipts
    assert bundle.dependency_receipts != donor.required_dependencies
    assert bundle_proof_allows_reuse(bundle, authoritative_receipt) is True


def test_dependency_receipt_tampering_invalidates_bundle_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    donor = _donor(required_dependencies=("geckolib",))
    monkeypatch.setattr(
        source_transplant,
        "materialize_pinned_donor",
        lambda *args, **kwargs: _materialized_payload(),
    )
    diagnostic_receipt = reuse_proof.execute_reuse_proof(
        donor,
        target_workspace=tmp_path,
        target_context={
            "loader": "fabric",
            "minecraft_version": "1.21.1",
            "java_version": 21,
            "target_package": "ai.minecraft.generated.mod",
            "target_modid": "generated_mod",
        },
        compile_checker=lambda *_: True,
    )
    authoritative_receipt = _authoritative_compile_receipt(diagnostic_receipt, donor)
    protected_artifacts = authoritative_receipt.contract["protected_artifacts"]
    bundle = ReusableArtifactBundle.from_donor_slice(
        donor,
        proof_receipt=authoritative_receipt,
        protected_artifacts=protected_artifacts,
    )
    assert bundle_proof_allows_reuse(bundle, authoritative_receipt) is True

    tampered = replace(bundle, dependency_receipts=())
    assert bundle_proof_allows_reuse(tampered, authoritative_receipt) is False


def test_unresolved_declared_dependency_cannot_become_verified_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    donor = _donor(
        required_dependencies=("com.example:unknown-worker6-library:1.0.0",)
    )
    monkeypatch.setattr(
        source_transplant,
        "materialize_pinned_donor",
        lambda *args, **kwargs: _materialized_payload(),
    )

    receipt = reuse_proof.execute_reuse_proof(
        donor,
        target_workspace=tmp_path,
        target_context={
            "loader": "fabric",
            "minecraft_version": "1.21.1",
            "java_version": 21,
            "target_package": "ai.minecraft.generated.mod",
            "target_modid": "generated_mod",
        },
        compile_checker=lambda *_: True,
    )

    assert ProofLevel.from_value(receipt.proof_level).is_verified() is False
    assert receipt.compile_passed is False
    assert len(receipt.dependency_receipts) == 1
    assert receipt.dependency_receipts[0]["is_resolved"] is False
    assert any("unknown-worker6-library" in value for value in receipt.unresolved_symbols)
