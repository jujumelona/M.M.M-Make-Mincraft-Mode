from __future__ import annotations

import json

from minecraft_mod_ai import planner_incremental_repair_contract as incremental
from minecraft_mod_ai.planner_checkpoint_journal_contract import install


def _batch(index: int) -> dict[str, object]:
    return {
        "batch_id": f"batch_{index}",
        "scope": f"scope {index}",
        "depends_on_batches": [],
        "deliverables": [f"d_{index}"],
        "exports": [f"export_{index}"],
    }


def test_checkpoint_journal_round_trips_and_does_not_rewrite_pending_queue(tmp_path) -> None:
    install(incremental)
    path = tmp_path / "planner.json"
    queue = [_batch(index) for index in range(5)]

    first = {
        "saved_batches": [],
        "pending_batches": list(queue),
        "pending_patch": None,
        "page_complete": False,
        "page_next_cursor": "next",
        "status": "collecting",
    }
    incremental._save_checkpoint(path, first)

    pending_path = path.with_name(path.name + ".pending.jsonl")
    events_path = path.with_name(path.name + ".events.jsonl")
    assert pending_path.is_file()
    initial_pending_bytes = pending_path.read_bytes()
    assert len(initial_pending_bytes.splitlines()) == 5

    second = {
        **first,
        "saved_batches": [queue[0]],
        "pending_batches": queue[1:],
    }
    incremental._save_checkpoint(path, second)

    third = {
        **second,
        "saved_batches": [queue[0], queue[1]],
        "pending_batches": queue[2:],
    }
    incremental._save_checkpoint(path, third)

    # The original pending payload is immutable; progress is append-only events.
    assert pending_path.read_bytes() == initial_pending_bytes
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 2

    meta = json.loads(path.read_text(encoding="utf-8"))
    assert "saved_batches" not in meta
    assert "pending_batches" not in meta
    assert meta["accepted_count"] == 2
    assert meta["pending_remaining"] == 3

    loaded = incremental._load_checkpoint(path)
    assert loaded["saved_batches"] == [queue[0], queue[1]]
    assert loaded["pending_batches"] == queue[2:]
    assert loaded["status"] == "collecting"


def test_checkpoint_patch_metadata_changes_without_rewriting_large_lists(tmp_path) -> None:
    install(incremental)
    path = tmp_path / "planner.json"
    queue = [_batch(index) for index in range(3)]
    state = {
        "saved_batches": [],
        "pending_batches": queue,
        "pending_patch": None,
        "status": "collecting",
    }
    incremental._save_checkpoint(path, state)
    pending_path = path.with_name(path.name + ".pending.jsonl")
    before = pending_path.read_bytes()

    state["pending_patch"] = {
        "target_fingerprint": "abc",
        "round": 3,
        "current_value": queue[0],
        "validation_error": "bad field",
    }
    state["status"] = "patching"
    incremental._save_checkpoint(path, state)

    assert pending_path.read_bytes() == before
    loaded = incremental._load_checkpoint(path)
    assert loaded["pending_batches"] == queue
    assert loaded["pending_patch"]["round"] == 3
    assert loaded["status"] == "patching"
