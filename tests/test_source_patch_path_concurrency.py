from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from minecraft_mod_ai.project_write_lock import (
    _state_for,
    project_path_write_locks,
    project_write_lock,
)
from minecraft_mod_ai.source_patch import (
    SourcePatchError,
    TransactionalSourcePatcher,
    sha256_bytes,
)


def _replace(path: str, before: str, after: str) -> dict[str, str]:
    return {
        "operation": "replace",
        "path": path,
        "expected_sha256": sha256_bytes(before.encode("utf-8")),
        "content": after,
    }


def test_disjoint_source_transactions_commit_concurrently(tmp_path, monkeypatch):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a0", encoding="utf-8")
    second.write_text("b0", encoding="utf-8")

    import minecraft_mod_ai.source_patch as source_patch

    original_commit = source_patch._commit_staged_path
    entered = Barrier(2)
    release = Barrier(3)

    def coordinated_commit(path, after):
        entered.wait(timeout=2)
        release.wait(timeout=2)
        return original_commit(path, after)

    monkeypatch.setattr(source_patch, "_commit_staged_path", coordinated_commit)
    patcher = TransactionalSourcePatcher(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(patcher.apply, [_replace("a.txt", "a0", "a1")]),
            pool.submit(patcher.apply, [_replace("b.txt", "b0", "b1")]),
        )
        release.wait(timeout=2)
        receipts = [future.result(timeout=2) for future in futures]

    assert [receipt["status"] for receipt in receipts] == ["APPLIED", "APPLIED"]
    assert first.read_text(encoding="utf-8") == "a1"
    assert second.read_text(encoding="utf-8") == "b1"


def test_overlapping_source_transactions_serialize_before_validation(tmp_path, monkeypatch):
    target = tmp_path / "shared.txt"
    target.write_text("old", encoding="utf-8")

    import minecraft_mod_ai.source_patch as source_patch

    original_commit = source_patch._commit_staged_path
    first_commit_entered = Event()
    release_first = Event()
    commit_calls = 0

    def blocked_commit(path, after):
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            first_commit_entered.set()
            assert release_first.wait(timeout=2)
        return original_commit(path, after)

    monkeypatch.setattr(source_patch, "_commit_staged_path", blocked_commit)
    patcher = TransactionalSourcePatcher(tmp_path)
    operation = _replace("shared.txt", "old", "first")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(patcher.apply, [operation])
        assert first_commit_entered.wait(timeout=2)
        second = pool.submit(
            patcher.apply,
            [_replace("shared.txt", "old", "second")],
        )
        assert not second.done()
        release_first.set()
        assert first.result(timeout=2)["status"] == "APPLIED"
        with pytest.raises(SourcePatchError, match="SHA-256 precondition failed"):
            second.result(timeout=2)

    assert commit_calls == 1
    assert target.read_text(encoding="utf-8") == "first"


def test_coarse_project_write_excludes_path_transaction(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    import minecraft_mod_ai.source_patch as source_patch

    original_commit = source_patch._commit_staged_path
    commit_entered = Event()

    def observed_commit(path, after):
        commit_entered.set()
        return original_commit(path, after)

    monkeypatch.setattr(source_patch, "_commit_staged_path", observed_commit)
    patcher = TransactionalSourcePatcher(tmp_path)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with project_write_lock(tmp_path):
            future = pool.submit(
                patcher.apply,
                [_replace("target.txt", "old", "new")],
            )
            assert not commit_entered.wait(timeout=0.05)
        assert future.result(timeout=2)["status"] == "APPLIED"

    assert commit_entered.is_set()
    assert target.read_text(encoding="utf-8") == "new"


def test_waiting_coarse_writer_preempts_new_scoped_transactions(tmp_path):
    first_entered = Event()
    release_first = Event()
    writer_entered = Event()
    release_writer = Event()
    second_entered = Event()

    def first_scoped():
        with project_path_write_locks(tmp_path, ("a.txt",)):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def coarse_writer():
        with project_write_lock(tmp_path):
            writer_entered.set()
            assert release_writer.wait(timeout=2)

    def second_scoped():
        with project_path_write_locks(tmp_path, ("b.txt",)):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=3) as pool:
        first_future = pool.submit(first_scoped)
        assert first_entered.wait(timeout=2)
        writer_future = pool.submit(coarse_writer)

        state = _state_for(tmp_path)
        with state.condition:
            assert state.condition.wait_for(
                lambda: state.waiting_writers == 1,
                timeout=2,
            )

        second_future = pool.submit(second_scoped)
        assert not second_entered.wait(timeout=0.05)
        release_first.set()
        assert writer_entered.wait(timeout=2)
        assert not second_entered.wait(timeout=0.05)
        release_writer.set()

        first_future.result(timeout=2)
        writer_future.result(timeout=2)
        second_future.result(timeout=2)

    assert second_entered.is_set()


def test_multi_path_transactions_use_deadlock_free_canonical_lock_order(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MMM_SOURCE_PATCH_WORKERS", "1")
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a0", encoding="utf-8")
    b.write_text("b0", encoding="utf-8")
    patcher = TransactionalSourcePatcher(tmp_path)

    first = [
        _replace("a.txt", "a0", "a1"),
        _replace("b.txt", "b0", "b1"),
    ]
    second = [
        _replace("b.txt", "b1", "b2"),
        _replace("a.txt", "a1", "a2"),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(patcher.apply, first)
        assert first_future.result(timeout=2)["status"] == "APPLIED"
        second_future = pool.submit(patcher.apply, second)
        assert second_future.result(timeout=2)["status"] == "APPLIED"

    assert a.read_text(encoding="utf-8") == "a2"
    assert b.read_text(encoding="utf-8") == "b2"
