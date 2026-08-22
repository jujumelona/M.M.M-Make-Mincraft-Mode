from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import source_patch as source_patch_module
from minecraft_mod_ai.causal_state_ledger import CausalStateLedger
from minecraft_mod_ai.source_patch import SourcePatchError, TransactionalSourcePatcher, sha256_bytes


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _rag_ok() -> dict:
    return {
        "role": "tool",
        "name": "search_code_rag",
        "content": {
            "ok": True,
            "result": {
                "hits": [{"path": "src/main/java/Test.java", "score": 1.0}],
                "receipt": {
                    "result_count": 1,
                    "coverage_score": 1.0,
                    "relevance_score": 1.0,
                },
            },
        },
    }


def _failed_patch(impact: str | None, *, error_type: str = "SourcePatchError") -> dict:
    suffix = f" [workspace_impact={impact}]" if impact else ""
    return {
        "role": "tool",
        "name": "apply_source_patch",
        "content": {"ok": False, "error": f"{error_type}: rejected{suffix}"},
    }


def _resolve(ledger: CausalStateLedger, messages: list[dict]) -> frozenset[str]:
    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    return ledger.resolve(
        messages,
        schemas,
        require_fresh_evidence=True,
        query_fn=lambda _messages: "repair project",
    ).state


def test_safe_failed_mutations_preserve_ready_evidence() -> None:
    ledger = CausalStateLedger()
    messages = [{"role": "user", "content": "repair project"}, _rag_ok()]
    ready = _resolve(ledger, messages)
    assert {"code_evidence", "evidence_ready"}.issubset(ready)

    messages = [*messages, _failed_patch("unchanged")]
    unchanged = _resolve(ledger, messages)
    assert {"code_evidence", "evidence_ready"}.issubset(unchanged)

    messages = [*messages, _failed_patch("rolled_back")]
    rolled_back = _resolve(ledger, messages)
    assert {"code_evidence", "evidence_ready"}.issubset(rolled_back)

    messages = [*messages, _failed_patch(None, error_type="SpecValidationError")]
    policy_rejected = _resolve(ledger, messages)
    assert {"code_evidence", "evidence_ready"}.issubset(policy_rejected)


def test_drift_and_uncertain_failures_invalidate_snapshot_evidence() -> None:
    for impact in ("drift", "uncertain"):
        ledger = CausalStateLedger()
        messages = [{"role": "user", "content": "repair project"}, _rag_ok()]
        assert "evidence_ready" in _resolve(ledger, messages)
        stale = _resolve(ledger, [*messages, _failed_patch(impact)])
        assert "code_evidence" not in stale
        assert "evidence_ready" not in stale
        assert "project_observed" in stale


def test_unknown_failed_mutation_remains_fail_closed() -> None:
    ledger = CausalStateLedger()
    messages = [{"role": "user", "content": "repair project"}, _rag_ok()]
    stale = _resolve(ledger, [*messages, _failed_patch(None)])
    assert "code_evidence" not in stale
    assert "evidence_ready" not in stale


def test_source_patch_sha_mismatch_reports_workspace_drift(tmp_path: Path) -> None:
    target = tmp_path / "src/main/java/Test.java"
    target.parent.mkdir(parents=True)
    target.write_text("class Test {}\n", encoding="utf-8")
    patcher = TransactionalSourcePatcher(tmp_path)

    with pytest.raises(SourcePatchError) as caught:
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/Test.java",
                    "expected_sha256": "sha256:" + "0" * 64,
                    "content": "class Test { int x; }\n",
                }
            ]
        )

    assert caught.value.workspace_impact == "drift"
    assert "[workspace_impact=drift]" in str(caught.value)
    assert target.read_text(encoding="utf-8") == "class Test {}\n"


def test_source_patch_prewrite_rejection_reports_unchanged(tmp_path: Path) -> None:
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError) as caught:
        patcher.apply([])
    assert caught.value.workspace_impact == "unchanged"


def test_source_patch_commit_failure_reports_successful_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "src/main/java/First.java"
    second = tmp_path / "src/main/java/Second.java"
    first.parent.mkdir(parents=True)
    first.write_text("class First {}\n", encoding="utf-8")
    second.write_text("class Second {}\n", encoding="utf-8")
    before_first = first.read_bytes()
    before_second = second.read_bytes()
    original_commit = source_patch_module._commit_staged_path
    calls = 0

    def fail_second(path: Path, after: bytes | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic commit failure")
        original_commit(path, after)

    monkeypatch.setenv("MMM_SOURCE_PATCH_WORKERS", "1")
    monkeypatch.setattr(source_patch_module, "_commit_staged_path", fail_second)
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError) as caught:
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/First.java",
                    "expected_sha256": sha256_bytes(before_first),
                    "content": "class First { int x; }\n",
                },
                {
                    "operation": "replace",
                    "path": "src/main/java/Second.java",
                    "expected_sha256": sha256_bytes(before_second),
                    "content": "class Second { int y; }\n",
                },
            ]
        )

    assert caught.value.workspace_impact == "rolled_back"
    assert first.read_bytes() == before_first
    assert second.read_bytes() == before_second


def test_source_patch_failed_rollback_reports_uncertain_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "src/main/java/First.java"
    second = tmp_path / "src/main/java/Second.java"
    first.parent.mkdir(parents=True)
    first.write_text("class First {}\n", encoding="utf-8")
    second.write_text("class Second {}\n", encoding="utf-8")
    before_first = first.read_bytes()
    before_second = second.read_bytes()
    original_commit = source_patch_module._commit_staged_path
    original_write_bytes = Path.write_bytes
    calls = 0

    def fail_second(path: Path, after: bytes | None) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic commit failure")
        original_commit(path, after)

    def fail_first_restore(path: Path, data: bytes) -> int:
        if path == first and data == before_first:
            raise OSError("synthetic rollback failure")
        return original_write_bytes(path, data)

    monkeypatch.setenv("MMM_SOURCE_PATCH_WORKERS", "1")
    monkeypatch.setattr(source_patch_module, "_commit_staged_path", fail_second)
    monkeypatch.setattr(Path, "write_bytes", fail_first_restore)
    patcher = TransactionalSourcePatcher(tmp_path)
    with pytest.raises(SourcePatchError) as caught:
        patcher.apply(
            [
                {
                    "operation": "replace",
                    "path": "src/main/java/First.java",
                    "expected_sha256": sha256_bytes(before_first),
                    "content": "class First { int x; }\n",
                },
                {
                    "operation": "replace",
                    "path": "src/main/java/Second.java",
                    "expected_sha256": sha256_bytes(before_second),
                    "content": "class Second { int y; }\n",
                },
            ]
        )

    assert caught.value.workspace_impact == "uncertain"
    assert "[workspace_impact=uncertain]" in str(caught.value)
