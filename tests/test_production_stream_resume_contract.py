from __future__ import annotations

import json

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai import production_stream_efficiency_contract as stream
from minecraft_mod_ai.production_stream_resume_contract import install as install_resume


def _module(module_id: str) -> dict[str, object]:
    return {
        "module_id": module_id,
        "kind": "item",
        "config": {},
        "depends_on": [],
        "required_gates": [],
        "implements_deliverables": [module_id],
    }


def test_saved_truncated_stream_is_salvaged_before_any_new_page_decode(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    request = {
        "current_target_deliverable": "alpha",
        "current_target_deliverables": ["alpha", "beta"],
        "remaining_deliverables": ["alpha", "beta"],
        "total_remaining": 2,
        "contract": dict(complete_planner._PRODUCTION_PAGE_CONTRACT),
    }
    stage = "production batch 'resume_stream' page"
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
    stream.install(complete_planner)
    install_resume(complete_planner)
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
