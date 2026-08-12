from __future__ import annotations

import json

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai.production_stream_efficiency_contract import install


def _module(module_id: str) -> dict[str, object]:
    return {
        "module_id": module_id,
        "kind": "item",
        "config": {},
        "depends_on": [],
        "required_gates": [],
        "implements_deliverables": [module_id],
    }


def _page(*module_ids: str) -> dict[str, object]:
    return {
        "modules": [_module(module_id) for module_id in module_ids],
        "assets": [],
        "audio": [],
        "acceptance_tests": [],
        "completed_deliverables": list(module_ids),
        "complete": True,
        "next_cursor": "",
    }


def _request(*targets: str) -> dict[str, object]:
    return {
        "current_target_deliverable": targets[0],
        "current_target_deliverables": list(targets),
        "remaining_deliverables": list(targets),
        "total_remaining": len(targets),
        "contract": dict(complete_planner._PRODUCTION_PAGE_CONTRACT),
    }


class _Router:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, *, media_paths=(), response_format="text"):
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": media_paths,
                "response_format": response_format,
            }
        )
        if not self.outputs:
            raise AssertionError("unexpected extra planner decode")
        value = self.outputs.pop(0)
        return value() if callable(value) else value


def _generate(router: _Router, request: dict[str, object]) -> dict[str, object]:
    install(complete_planner)
    return complete_planner._generate_json_page_with_repair(
        router,
        system_prompt="PRODUCTION",
        request=request,
        media_paths=(),
        expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),
        stage="production batch 'stream_test' page",
    )


def test_normal_production_decode_keeps_all_host_targets_without_hidden_width_cap(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    targets = ("one", "two", "three", "four", "five")
    router = _Router([json.dumps(_page(*targets))])

    page = _generate(router, _request(*targets))

    assert page["completed_deliverables"] == list(targets)
    assert len(router.calls) == 1
    sent = json.loads(router.calls[0]["messages"][-1]["content"])
    assert sent["current_target_deliverables"] == list(targets)
    assert sent["remaining_deliverables"] == list(targets)
    assert "NO fixed item/deliverable width" in router.calls[0]["messages"][0]["content"]


def test_truncated_outer_page_saves_complete_sibling_and_repairs_only_tail_child(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    first = _module("alpha")
    truncated = (
        '{"modules":['
        + json.dumps(first, separators=(",", ":"))
        + ',{"module_id":"beta","kind":"item","config":{"unfinished":'
    )

    router = _Router([truncated])

    def repair_output():
        call = router.calls[-1]
        request = json.loads(call["messages"][-1]["content"])
        return json.dumps(
            {
                "target_fingerprint": request["target_fingerprint"],
                "replacement": _module("beta"),
            }
        )

    router.outputs.append(repair_output)
    page = _generate(router, _request("alpha", "beta"))

    assert [item["module_id"] for item in page["modules"]] == ["alpha", "beta"]
    assert page["completed_deliverables"] == ["alpha", "beta"]
    assert page["complete"] is True
    assert len(router.calls) == 2
    assert "exactly ONE truncated" in router.calls[1]["messages"][0]["content"]

    stream_files = list((tmp_path / "production_pages").glob("*.stream.jsonl"))
    assert len(stream_files) == 1
    saved = json.loads(stream_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert saved["text"] == truncated


def test_multiple_complete_production_json_pages_are_consumed_in_one_decode(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    text = json.dumps(_page("alpha")) + "\n" + json.dumps(_page("beta"))
    router = _Router([text])

    page = _generate(router, _request("alpha", "beta"))

    assert len(router.calls) == 1
    assert [item["module_id"] for item in page["modules"]] == ["alpha", "beta"]
    assert page["completed_deliverables"] == ["alpha", "beta"]
    assert page["complete"] is True
    assert page["next_cursor"] == ""
