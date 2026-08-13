from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai import production_stream_efficiency_contract as stream


def _module(module_id: str) -> dict[str, object]:
    return {
        "module_id": module_id,
        "kind": "item",
        "config": {},
        "depends_on": [],
        "required_gates": [],
        "implements_deliverables": [module_id],
    }


def _request() -> dict[str, object]:
    return {
        "current_target_deliverable": "alpha",
        "current_target_deliverables": ["alpha", "beta"],
        "remaining_deliverables": ["alpha", "beta"],
        "total_remaining": 2,
        "contract": dict(complete_planner._PRODUCTION_PAGE_CONTRACT),
    }


def _saved_truncated_stream(tmp_path, *, stage: str) -> tuple[dict[str, object], str]:
    request = _request()
    truncated = (
        '{"modules":['
        + json.dumps(_module("alpha"), separators=(",", ":"))
        + ',{"module_id":"beta","kind":"item","config":{"unfinished":'
    )
    stream._append_stream_event(
        stage=stage,
        request=request,
        round_index=0,
        text=truncated,
        diagnostic="simulated crash after stream fsync",
    )
    return request, truncated


def test_saved_truncated_stream_is_salvaged_before_any_new_page_decode(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    stage = "production batch 'resume_stream' page"
    request, _ = _saved_truncated_stream(tmp_path, stage=stage)

    class Router:
        def __init__(self) -> None:
            self.calls = []

        def generate_text(self, role, messages, *, media_paths=(), response_format="text"):
            self.calls.append(messages)
            user = json.loads(messages[-1]["content"])
            # The only allowed model call is repair of beta. A fresh production-page
            # decode would not contain target_fingerprint and makes this test fail.
            fingerprint = user["target_fingerprint"]
            return json.dumps(
                {
                    "target_fingerprint": fingerprint,
                    "replacement": _module("beta"),
                }
            )

    router = Router()
    page = complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="PRODUCTION FULL PAGE SHOULD NOT RUN",
        request=request,
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage=stage,
    )

    assert len(router.calls) == 1
    assert "exactly ONE truncated" in router.calls[0][0]["content"]
    assert [item["module_id"] for item in page["modules"]] == ["alpha", "beta"]
    assert page["completed_deliverables"] == ["alpha", "beta"]
    assert page["complete"] is True


def test_saved_stream_backend_failure_never_falls_through_to_full_page_decode(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    stage = "production batch 'resume_backend_down' page"
    request, truncated = _saved_truncated_stream(tmp_path, stage=stage)

    class DownRouter:
        def __init__(self) -> None:
            self.calls = []

        def generate_text(self, role, messages, *, media_paths=(), response_format="text"):
            self.calls.append(messages)
            # If recovery incorrectly falls through, this would be called again for a
            # full page. The contract requires the first backend failure to propagate.
            raise RuntimeError("backend down")

    router = DownRouter()

    with pytest.raises(RuntimeError, match="backend down"):
        complete_planner._generate_json_page_with_repair(
            router,
            system_prompt="FULL PAGE MUST NEVER RUN AFTER SAVED-STREAM FAILURE",
            request=request,
            media_paths=(),
            expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
            stage=stage,
        )

    assert len(router.calls) == 1
    assert "exactly ONE truncated" in router.calls[0][0]["content"]

    stream_path = stream._stream_event_path(stage, request)
    saved = json.loads(stream_path.read_text(encoding="utf-8").splitlines()[0])
    assert saved["text"] == truncated

    # The child repair checkpoint must also exist, proving the restart point is the
    # same beta fragment rather than an uncheckpointed page regeneration.
    repair_states = list(stream_path.parent.glob(stream_path.name.removesuffix(".stream.jsonl") + ".truncated-module-*.json"))
    assert repair_states
    state = json.loads(repair_states[0].read_text(encoding="utf-8"))
    assert state["status"] == "repairing"
    assert state["fragment"].startswith('{"module_id":"beta"')
