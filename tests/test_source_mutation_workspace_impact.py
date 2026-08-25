from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import source_patch as source_patch_module
from minecraft_mod_ai.source_patch import SourcePatchError, TransactionalSourcePatcher, sha256_bytes


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
