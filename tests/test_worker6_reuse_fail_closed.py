from __future__ import annotations

import hashlib

import pytest

import minecraft_mod_ai.reuse_proof_executor as reuse_proof
import minecraft_mod_ai.source_transplant as source_transplant
from minecraft_mod_ai.proof_level import ProofLevel, validate_proof_transition
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice, SourceTransplantError


def _donor(
    *,
    repository: str = "example/worker6-donor",
    license_id: str = "MIT",
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
        required_dependencies=(),
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
