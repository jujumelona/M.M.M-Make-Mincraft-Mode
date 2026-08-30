from __future__ import annotations

import hashlib

import pytest

import minecraft_mod_ai.reuse_proof_executor as reuse_proof
import minecraft_mod_ai.source_transplant as source_transplant
from minecraft_mod_ai.proof_level import ProofLevel
from minecraft_mod_ai.source_transplant import DonorFile, DonorSlice, SourceTransplantError


def _donor(repository: str = "example/worker6") -> DonorSlice:
    payload = b"package donor; public class BossEntity {}\n"
    return DonorSlice(
        capability="boss.entity", repository=repository, commit_sha="a" * 40,
        license_id="MIT", source_url=f"https://github.com/{repository}",
        target_compatibility="metadata_exact",
        files=(DonorFile(path="src/main/java/donor/BossEntity.java", blob_sha="b" * 40,
            sha256="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
            symbols=("BossEntity",)),),
        seed_files=("src/main/java/donor/BossEntity.java",), source_symbols=("BossEntity",),
        required_dependencies=(), donor_tests=(), confidence=0.9, closure_complete=True,
    )


def _partial(candidate: DonorSlice, verified: int, residual: int) -> reuse_proof.ReuseProofReceipt:
    return reuse_proof.ReuseProofReceipt(
        candidate_id=f"{candidate.repository}@{candidate.commit_sha}", capability=candidate.capability,
        commit_sha=candidate.commit_sha, closure_hash="sha256:test",
        proof_level=ProofLevel.PARTIAL_REUSE.value, compile_passed=False, tests_passed=False,
        unresolved_symbols=(), missing_resources=(), adaptations_applied=(),
        verified_capabilities=(), residual_capabilities=(candidate.capability,),
        verified_artifacts=tuple(f"verified-{i}.java" for i in range(verified)),
        residual_artifacts=tuple(f"residual-{i}.java" for i in range(residual)),
        verified_symbols=tuple(f"Verified{i}" for i in range(verified)),
    )


def test_materialization_failure_never_uses_mock_source_for_checker(monkeypatch, tmp_path):
    checker_called = False
    def fail_materialize(*args, **kwargs):
        raise SourceTransplantError("donor unavailable")
    def checker(*args, **kwargs):
        nonlocal checker_called
        checker_called = True
        return True
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", fail_materialize)
    receipt = reuse_proof.execute_reuse_proof(
        _donor(), target_workspace=tmp_path, target_context={}, compile_checker=checker
    )
    assert checker_called is False
    assert receipt.compile_passed is False
    assert not ProofLevel.from_value(receipt.proof_level).allows_reuse()


def test_compile_checker_programming_error_propagates(monkeypatch, tmp_path):
    payload = b"package donor; public class BossEntity {}\n"
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor",
        lambda *args, **kwargs: {"src/main/java/donor/BossEntity.java": payload})
    def explode(*args, **kwargs):
        raise RuntimeError("checker programming error")
    with pytest.raises(RuntimeError, match="checker programming error"):
        reuse_proof.execute_reuse_proof(
            _donor(), target_workspace=tmp_path, target_context={}, compile_checker=explode
        )


def test_fallback_selects_strongest_partial_receipt(monkeypatch, tmp_path):
    weak = _donor("example/weak")
    strong = _donor("example/strong")
    receipts = iter((_partial(weak, 1, 4), _partial(strong, 3, 1)))
    monkeypatch.setattr(reuse_proof, "execute_reuse_proof",
        lambda *args, **kwargs: next(receipts))
    selected, all_receipts = reuse_proof.execute_candidate_fallback_loop(
        (weak, strong), "boss.entity", target_workspace=tmp_path, target_context={}
    )
    assert selected is strong
    assert len(all_receipts) == 2


def test_transient_repository_snapshot_failure_is_not_negative_cached():
    source_transplant._SNAPSHOT_CACHE.clear()
    source_transplant._SNAPSHOT_INFLIGHT.clear()
    calls = 0
    class Discovery:
        github_token = ""
        def inspect_github_repository(self, repository):
            nonlocal calls
            calls += 1
            raise SourceTransplantError("transient")
    discovery = Discovery()
    assert source_transplant._repository_snapshot("example/transient", discovery) is None
    assert source_transplant._repository_snapshot("example/transient", discovery) is None
    assert calls == 2


def test_unexpected_snapshot_programming_error_is_not_hidden():
    source_transplant._SNAPSHOT_CACHE.clear()
    source_transplant._SNAPSHOT_INFLIGHT.clear()
    class Discovery:
        github_token = ""
        def inspect_github_repository(self, repository):
            raise RuntimeError("snapshot programming error")
    with pytest.raises(RuntimeError, match="snapshot programming error"):
        source_transplant._repository_snapshot("example/programming", Discovery())
