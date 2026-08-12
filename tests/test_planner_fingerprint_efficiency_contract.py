from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from minecraft_mod_ai import planner_checkpoint_journal_contract as journal


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _plain_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fake_incremental(tmp_path: Path, fingerprint_calls: list[Any]) -> SimpleNamespace:
    def fingerprint(value: Any) -> str:
        fingerprint_calls.append(value)
        return _plain_fingerprint(value)

    def batch_identity(value: Any) -> str:
        if isinstance(value, dict):
            batch_id = str(value.get("batch_id", "")).strip()
            if batch_id:
                return "id:" + batch_id
        return "sha256:" + fingerprint(value)

    def accepted_batch_ids(saved: list[Any]) -> list[str]:
        return [
            str(value.get("batch_id", "")).strip()
            for value in saved
            if isinstance(value, dict) and str(value.get("batch_id", "")).strip()
        ]

    def merge_saved(saved: list[Any], incoming: list[Any]) -> None:
        known = {batch_identity(value) for value in saved}
        for value in incoming:
            identity = batch_identity(value)
            if identity not in known:
                saved.append(value)
                known.add(identity)

    def checkpoint_path(stage: str, request: Any) -> Path:
        digest = _plain_fingerprint({"stage": stage, "request": request})
        safe_stage = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in stage
        ).strip("_")[:60] or "planner"
        return tmp_path / f"{safe_stage}-{digest[:20]}.json"

    def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
        path.write_text(
            json.dumps({"version": 2, **state}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_checkpoint(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("version") != 2:
            return {}
        return value

    return SimpleNamespace(
        _CHECKPOINT_VERSION=2,
        _canonical_bytes=_canonical,
        _checkpoint_root=lambda: tmp_path,
        _fingerprint=fingerprint,
        _batch_identity=batch_identity,
        _accepted_batch_ids=accepted_batch_ids,
        _merge_saved_batches=merge_saved,
        _checkpoint_path=checkpoint_path,
        _save_checkpoint=save_checkpoint,
        _load_checkpoint=load_checkpoint,
    )


def _batch(index: int) -> dict[str, Any]:
    return {
        "batch_id": f"batch_{index}",
        "scope": "x" * 2048,
        "depends_on_batches": [],
        "deliverables": [f"deliverable_{index}"],
        "exports": [f"export_{index}"],
    }


def test_request_digest_is_reused_and_checkpoint_path_is_legacy_compatible(
    tmp_path: Path,
) -> None:
    calls: list[Any] = []
    module = _fake_incremental(tmp_path, calls)
    stage = "large request fingerprint"
    request = {
        "contract": {"field": "x" * 200_000},
        "known_batch_catalog": {"count": 10_000, "recent_ids": []},
    }
    expected = module._checkpoint_path(stage, request)

    journal.install(module)
    calls.clear()
    actual = module._checkpoint_path(stage, request)

    assert actual == expected
    expected_request_digest = _plain_fingerprint(request)
    for _ in range(20):
        assert module._fingerprint(request) == expected_request_digest
    # checkpoint_path canonicalizes the request once directly; the repeated
    # _fingerprint(request) calls must never fall through to the old serializer.
    assert calls == []
    assert getattr(module._checkpoint_path, "_mmm_request_fingerprint_cache", False)


def test_saved_prefix_hash_and_identity_tracking_are_incremental(tmp_path: Path) -> None:
    calls: list[Any] = []
    module = _fake_incremental(tmp_path, calls)
    journal.install(module)

    saved = [_batch(index) for index in range(500)]
    # Registers the immutable accepted prefix once. This first construction is linear.
    assert len(module._accepted_batch_ids(saved)) == 500
    initial_build_calls = len(calls)
    assert initial_build_calls == 500
    first_digest = module._fingerprint(saved)
    assert len(first_digest) == 64

    calls.clear()
    for index in range(500, 600):
        module._merge_saved_batches(saved, [_batch(index)])
        # Reading the current saved-prefix digest is O(1); it must not serialize the
        # previous 500+ accepted batches again.
        assert len(module._fingerprint(saved)) == 64

    assert len(saved) == 600
    assert len(calls) == 100
    assert all(isinstance(value, dict) and "batch_id" in value for value in calls)
    assert module._fingerprint(saved) != first_digest

    calls.clear()
    cursor_digest = module._fingerprint(
        {"saved": saved, "new": [_batch(600), _batch(601)]}
    )
    assert len(cursor_digest) == 64
    # host_resume hashes only the new page once, never the 600-item saved prefix.
    assert len(calls) == 1
    assert isinstance(calls[0], list)
    assert len(calls[0]) == 2
    assert getattr(module._merge_saved_batches, "_mmm_incremental_identity_cache", False)


def test_persisted_saved_digest_avoids_rehash_after_resume(tmp_path: Path) -> None:
    calls: list[Any] = []
    module = _fake_incremental(tmp_path, calls)
    journal.install(module)

    request = {"contract": {"large": "x" * 10_000}}
    path = module._checkpoint_path("resume digest", request)
    saved = [_batch(index) for index in range(80)]
    module._accepted_batch_ids(saved)
    expected_digest = module._fingerprint(saved)
    module._save_checkpoint(
        path,
        {
            "stage": "resume digest",
            "saved_batches": saved,
            "pending_batches": [],
            "status": "page_complete",
        },
    )

    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["saved_batches_digest_version"] == 1
    assert meta["saved_batches_digest_count"] == 80
    assert meta["saved_batches_digest"] == expected_digest

    calls.clear()
    result: dict[str, Any] = {}

    def resume_in_fresh_thread() -> None:
        state = module._load_checkpoint(path)
        resumed = list(state["saved_batches"])
        assert len(module._accepted_batch_ids(resumed)) == 80
        result["digest"] = module._fingerprint(resumed)

    thread = threading.Thread(target=resume_in_fresh_thread)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert result["digest"] == expected_digest
    # Persisted digest + count + endpoint identity restore the accumulator without
    # serializing any of the 80 historical batches again.
    assert calls == []
