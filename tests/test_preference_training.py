from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.preference_training import PreferenceCandidate, PreferenceTraceStore


def test_preference_trace_records_winner_and_exports_dpo(tmp_path: Path) -> None:
    store = PreferenceTraceStore(tmp_path / "preferences.jsonl")
    receipt = store.record(
        task="repair_patch_selection",
        prompt={"error": "cannot find symbol"},
        candidates=(
            PreferenceCandidate(
                candidate_id="a",
                response={"operations": [{"path": "A.java"}]},
                score=-2.0,
                verifier={"jdt_error_count": 2},
            ),
            PreferenceCandidate(
                candidate_id="b",
                response={"operations": [{"path": "A.java", "fix": True}]},
                score=1000.0,
                verifier={"jdt_error_count": 0},
            ),
        ),
        winner_index=1,
        metadata={"search_width": 2},
    )
    assert receipt["candidate_count"] == 2
    assert receipt["winner_id"] == "b"

    output = tmp_path / "dpo.jsonl"
    exported = store.export_dpo(output)
    assert exported["pairs"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["chosen"]["operations"][0]["fix"] is True
    assert row["rejected"]["operations"][0]["path"] == "A.java"
    assert row["metadata"]["chosen_score"] > row["metadata"]["rejected_score"]


def test_duplicate_preference_trace_is_idempotent(tmp_path: Path) -> None:
    store = PreferenceTraceStore(tmp_path / "preferences.jsonl")
    kwargs = {
        "task": "planner_selection",
        "prompt": "request",
        "candidates": (
            PreferenceCandidate("a", "A", 0.0, {}),
            PreferenceCandidate("b", "B", 1.0, {}),
        ),
        "winner_index": 1,
    }
    first = store.record(**kwargs)
    second = store.record(**kwargs)
    assert first["trace_id"] == second["trace_id"]
    assert len(store.path.read_text(encoding="utf-8").splitlines()) == 1
