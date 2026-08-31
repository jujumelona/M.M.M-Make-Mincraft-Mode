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
from minecraft_mod_ai.proof_level import ProofLevel
from minecraft_mod_ai.source_transplant import (
    DonorFile,
    DonorSlice,
    SourceTransplantError,
)


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



def test_invalid_commit_is_rejected_before_materialization(monkeypatch, tmp_path):
    donor = _donor()
    invalid = source_transplant.DonorSlice(
        **{**donor.__dict__, "commit_sha": "abc123"}
    )
    called = False
    def materialize(*args, **kwargs):
        nonlocal called
        called = True
        return {}
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", materialize)
    receipt = reuse_proof.execute_reuse_proof(
        invalid, target_workspace=tmp_path, target_context={}, compile_checker=lambda *_: True
    )
    assert called is False
    assert receipt.compile_passed is False
    assert receipt.proof_level == ProofLevel.DISCOVERED.value


def test_unsafe_manifest_path_is_rejected_before_io(monkeypatch, tmp_path):
    donor = _donor()
    bad_file = source_transplant.DonorFile(
        path="../escape.java", blob_sha="b" * 40,
        sha256=donor.files[0].sha256, size_bytes=donor.files[0].size_bytes,
        symbols=("BossEntity",),
    )
    unsafe = source_transplant.DonorSlice(
        **{**donor.__dict__, "files": (bad_file,)}
    )
    called = False
    def materialize(*args, **kwargs):
        nonlocal called
        called = True
        return {}
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", materialize)
    receipt = reuse_proof.execute_reuse_proof(
        unsafe, target_workspace=tmp_path, target_context={}, compile_checker=lambda *_: True
    )
    assert called is False
    assert receipt.compile_passed is False


def test_pinned_materialization_verifies_declared_size(monkeypatch):
    donor = _donor()
    wrong_size = source_transplant.DonorSlice(
        **{**donor.__dict__, "files": (
            source_transplant.DonorFile(
                path=donor.files[0].path, blob_sha=donor.files[0].blob_sha,
                sha256=donor.files[0].sha256, size_bytes=donor.files[0].size_bytes + 1,
                symbols=donor.files[0].symbols,
            ),
        )}
    )
    payload = b"package donor; public class BossEntity {}\n"
    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", lambda *args, **kwargs: payload)
    with pytest.raises(SourceTransplantError, match="size mismatch"):
        source_transplant.materialize_pinned_donor(
            wrong_size, discovery_client=type("D", (), {"_client": object()})()
        )


def test_reused_tree_sha_is_walked_for_each_prefix(monkeypatch):
    root = "1" * 40
    shared = "2" * 40
    commit = "a" * 40
    def fake_json(client, url, *, params=None):
        del client
        if "/git/commits/" in url:
            return {"tree": {"sha": root}}
        if url.endswith(root) and params == {"recursive": "1"}:
            return {"truncated": True, "tree": []}
        if url.endswith(root):
            return {"truncated": False, "tree": [
                {"path": "a", "sha": shared, "type": "tree"},
                {"path": "b", "sha": shared, "type": "tree"},
            ]}
        if url.endswith(shared):
            return {"truncated": False, "tree": [
                {"path": "Boss.java", "sha": "3" * 40, "type": "blob"}
            ]}
        raise AssertionError(url)
    monkeypatch.setattr(source_transplant, "_github_json", fake_json)
    entries = source_transplant._repository_tree_entries(object(), "example/repo", commit)
    assert {item["path"] for item in entries} == {"a/Boss.java", "b/Boss.java"}


def test_blob_cache_is_byte_bounded_lru(monkeypatch):
    source_transplant._BLOB_CACHE.clear()
    source_transplant._BLOB_CACHE_BYTES = 0
    monkeypatch.setenv("MMM_SOURCE_TRANSPLANT_SINGLE_BLOB_BYTE_BUDGET", str(64 * 1024))
    monkeypatch.setenv("MMM_SOURCE_TRANSPLANT_BLOB_CACHE_BYTE_BUDGET", str(128 * 1024))
    payloads = {
        "1" * 40: b"a" * 60_000,
        "2" * 40: b"b" * 60_000,
        "3" * 40: b"c" * 60_000,
    }
    import base64
    def fake_json(client, url, *, params=None):
        del client, params
        sha = url.rsplit("/", 1)[-1]
        return {"encoding": "base64", "content": base64.b64encode(payloads[sha]).decode()}
    monkeypatch.setattr(source_transplant, "_github_json", fake_json)
    for sha in payloads:
        source_transplant._fetch_blob_bytes(object(), "example/repo", sha)
    assert source_transplant._BLOB_CACHE_BYTES <= source_transplant._blob_cache_byte_budget()
    assert len(source_transplant._BLOB_CACHE) == 2
    assert ("example/repo", "1" * 40) not in source_transplant._BLOB_CACHE


def test_sandbox_destination_rejects_parent_escape(tmp_path):
    with pytest.raises(reuse_proof.ReuseTargetWorkspaceError):
        reuse_proof._sandbox_destination(tmp_path, "../escape.java")
