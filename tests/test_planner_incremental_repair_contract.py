from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai import planner_incremental_repair_contract as incremental
from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai.planner_strict_json_contract import install as install_strict
from minecraft_mod_ai.spec import SpecValidationError


class _Router:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role: str,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ) -> str:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": media_paths,
                "response_format": response_format,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return str(value)


def _request() -> dict[str, object]:
    return {
        "known_batch_catalog": {"count": 0, "sha256": "0" * 64, "recent_ids": []},
        "cursor": "",
        "contract": complete_planner._PRODUCTION_OUTLINE_CONTRACT,
    }


def _batch(
    batch_id: str,
    *,
    scope: str = "Implement this production area.",
    deliverables: list[str] | None = None,
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "scope": scope,
        "depends_on_batches": [],
        "deliverables": deliverables if deliverables is not None else [f"{batch_id}_done"],
        "exports": [f"{batch_id}_api"],
    }


def _outline(*batches: dict[str, object]) -> str:
    return json.dumps(
        {
            "production_batches": list(batches),
            "complete": True,
            "next_cursor": "",
        }
    )


def _field_patch(raw: object, **set_fields: object) -> str:
    return json.dumps(
        {
            "target_fingerprint": incremental._fingerprint(raw),
            "set_fields": set_fields,
            "delete_fields": [],
        }
    )


def _replacement_patch(raw: object, replacement: dict[str, object]) -> str:
    return json.dumps(
        {
            "target_fingerprint": incremental._fingerprint(raw),
            "replacement_batch": replacement,
        }
    )


def _run(router: _Router, *, stage: str) -> dict[str, object]:
    install_strict(runtime)
    return complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="ignored by scalable outline contract",
        request=_request(),
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_OUTLINE_CONTRACT),),
        stage=stage,
    )


def _only_checkpoint(tmp_path: Path) -> Path:
    files = [path for path in tmp_path.glob("*.json") if not path.name.endswith(".resolved.json")]
    assert len(files) == 1
    return files[0]


def test_invalid_batch_repairs_only_bad_field_and_preserves_sibling(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    good = _batch("core_runtime")
    bad = _batch("ui_runtime", deliverables=[])
    router = _Router(
        _outline(good, bad),
        _field_patch(bad, deliverables=["ui_runtime_done"]),
    )

    page = _run(router, stage="field patch sibling preservation")

    assert page["complete"] is True
    assert page["production_batches"][0] == good
    repaired = page["production_batches"][1]
    assert repaired["batch_id"] == bad["batch_id"]
    assert repaired["scope"] == bad["scope"]
    assert repaired["depends_on_batches"] == bad["depends_on_batches"]
    assert repaired["exports"] == bad["exports"]
    assert repaired["deliverables"] == ["ui_runtime_done"]
    assert len(router.calls) == 2
    patch_system = str(router.calls[1]["messages"][0]["content"])
    assert "field-level JSON patcher" in patch_system
    assert "DO NOT rewrite the whole batch" in patch_system
    patch_user = json.loads(str(router.calls[1]["messages"][1]["content"]))
    assert patch_user["current_batch"] == bad

    checkpoint = incremental._load_checkpoint(_only_checkpoint(tmp_path))
    assert checkpoint["status"] == "complete"
    assert checkpoint["pending_batches"] == []
    assert checkpoint["pending_patch"] is None
    assert checkpoint["saved_batches"] == [good, repaired]


def test_invalid_field_patch_escalates_to_one_complete_batch_regeneration(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    bad = _batch("broken_scope", scope="")
    repaired = _batch("broken_scope", scope="Implement the corrected scope.")
    router = _Router(
        _outline(bad),
        _field_patch(bad, scope=""),
        _replacement_patch(bad, repaired),
    )

    page = _run(router, stage="field patch then regenerate")

    assert page["complete"] is True
    assert page["production_batches"] == [repaired]
    assert len(router.calls) == 3
    assert "field-level JSON patcher" in str(router.calls[1]["messages"][0]["content"])
    assert "regenerate exactly ONE invalid production batch" in str(
        router.calls[2]["messages"][0]["content"]
    )
    second_request = json.loads(str(router.calls[2]["messages"][1]["content"]))
    assert second_request["repair_mode"] == "replacement"


def test_backend_failure_resumes_pending_patch_without_regenerating_outline(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    good = _batch("saved_core")
    bad = _batch("resume_ui", deliverables=[])
    first_router = _Router(_outline(good, bad), RuntimeError("server disconnected"))

    with pytest.raises(RuntimeError, match="server disconnected"):
        _run(first_router, stage="resume field patch")

    checkpoint_path = _only_checkpoint(tmp_path)
    interrupted = incremental._load_checkpoint(checkpoint_path)
    assert interrupted["saved_batches"] == [good]
    assert interrupted["pending_batches"] == [bad]
    assert interrupted["pending_patch"]["current_value"] == bad

    second_router = _Router(_field_patch(bad, deliverables=["resume_ui_done"]))
    page = _run(second_router, stage="resume field patch")

    assert page["complete"] is True
    assert [item["batch_id"] for item in page["production_batches"]] == [
        "saved_core",
        "resume_ui",
    ]
    assert page["production_batches"][1]["deliverables"] == ["resume_ui_done"]
    assert len(second_router.calls) == 1
    assert "field-level JSON patcher" in str(
        second_router.calls[0]["messages"][0]["content"]
    )


def test_duplicate_batch_id_is_patched_not_silently_dropped(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    first = _batch("duplicate_id")
    duplicate = _batch("duplicate_id", scope="A distinct second production area.")
    patch = _field_patch(duplicate, batch_id="distinct_second_area")
    router = _Router(_outline(first, duplicate), patch)

    page = _run(router, stage="duplicate id patch")

    assert [item["batch_id"] for item in page["production_batches"]] == [
        "duplicate_id",
        "distinct_second_area",
    ]
    assert page["production_batches"][1]["scope"] == duplicate["scope"]
    assert len(router.calls) == 2
    patch_request = json.loads(str(router.calls[1]["messages"][1]["content"]))
    assert "duplicate batch_id" in patch_request["validation_error"]


def test_identical_bad_patch_output_is_not_sent_or_consumed_forever(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    bad = _batch("broken_scope", scope="")
    repeated = _field_patch(bad, scope="")
    router = _Router(_outline(bad), repeated, repeated)

    with pytest.raises(SpecValidationError, match="repeated identical model output"):
        _run(router, stage="repeated batch patch output")

    # One outline call + exactly two distinct repair modes. No third repair call exists.
    assert len(router.calls) == 3


def test_broken_outline_stops_only_at_exact_request_response_fixed_point(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    router = _Router("{}", "{}", "{}")
    with pytest.raises(SpecValidationError, match="fixed point"):
        _run(router, stage="outline fixed point")
    assert len(router.calls) == 3


def test_outline_can_keep_repairing_beyond_two_generations(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    final = _batch("eventual_valid_outline")
    router = _Router(
        '{"diagnostic":1}',
        '{"diagnostic":2}',
        '{"diagnostic":3}',
        '{"diagnostic":4}',
        _outline(final),
    )
    page = _run(router, stage="long progressive outline")
    assert page["production_batches"] == [final]
    assert len(router.calls) == 5
