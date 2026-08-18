from __future__ import annotations

from minecraft_mod_ai import procedure_trace, trajectory_record_integrity
from minecraft_mod_ai.unified_trajectory_memory_contract import _repair_memory_rows


def test_repair_memory_adapter_uses_only_strong_v3_rows(monkeypatch) -> None:
    monkeypatch.setattr(trajectory_record_integrity, "record_strong_skill_eligible", lambda row: row.get("outcome") == "SUCCESS")
    monkeypatch.setattr(procedure_trace, "sequence_actions", lambda procedure: tuple(procedure.get("actions", ())) if procedure else ())
    rows = _repair_memory_rows(
        "registry compile error",
        [
            {"task_class": "repair", "outcome": "SUCCESS", "trajectory_id": "sha256:good", "error_signature": "registry compile error", "verified_facts": {"build_status": "PASS"}, "procedure": {"actions": ["search_code_rag", "apply_source_patch", "run_gradle_build"]}},
            {"task_class": "repair", "outcome": "FAIL", "trajectory_id": "sha256:bad", "error_signature": "registry compile error", "verified_facts": {"build_status": "FAIL"}, "procedure": {"actions": ["apply_source_patch"]}},
        ],
    )
    assert len(rows) == 1
    assert rows[0]["signature_sha256"] == "sha256:good"
    assert rows[0]["source"] == "mmm/verified-trajectory-v3"
    assert [item["operation"] for item in rows[0]["repair_pattern"]] == ["search_code_rag", "apply_source_patch", "run_gradle_build"]
