from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai import planner_production_page_contract as contract


def _module(module_id: str) -> dict[str, object]:
    return {
        "module_id": module_id,
        "kind": "custom_java",
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }


def _run(monkeypatch, deliverables: tuple[str, ...], pages: list[dict[str, object]]):
    contract.install(complete_planner)
    calls: list[dict[str, object]] = []

    def generate(_router, *, system_prompt, request, **_kwargs):
        calls.append({"system_prompt": system_prompt, "request": request})
        return pages[len(calls) - 1]

    monkeypatch.setattr(complete_planner, "_generate_json_page_with_repair", generate)
    planner = complete_planner.CompleteGameDesignPlanner(SimpleNamespace())
    batch = complete_planner._ProductionBatch(
        batch_id="adaptive_batch",
        scope="implement all requested work",
        depends_on_batches=(),
        deliverables=deliverables,
        exports=(),
    )
    parts = complete_planner._ProductionParts([], [], [], [])
    planner._expand_one_production_batch(
        batch=batch,
        parts=parts,
        module_catalog=complete_planner._ModuleCatalog(),
        asset_catalog=complete_planner._ModuleCatalog(),
        audio_catalog=complete_planner._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={"context": "x"},
        planning_receipt={"sha256": "x"},
        media_paths=(),
    )
    return calls, parts


def test_all_five_deliverables_can_finish_in_one_model_call(monkeypatch) -> None:
    deliverables = tuple(f"deliverable_{index}" for index in range(5))
    page = {
        "modules": [_module(f"module_{index}") for index in range(5)],
        "assets": [],
        "audio": [],
        "acceptance_tests": [],
        "completed_deliverables": list(deliverables),
        "complete": True,
        "next_cursor": "",
    }

    calls, parts = _run(monkeypatch, deliverables, [page])

    assert len(calls) == 1
    request = calls[0]["request"]
    assert request["current_target_deliverables"] == list(deliverables)
    assert request["remaining_deliverables"] == list(deliverables)
    assert len(parts.modules) == 5
    prompt = str(calls[0]["system_prompt"])
    assert "NO fixed deliverable count" in prompt
    assert "one module per deliverable" in prompt
    assert "Do NOT force" in prompt


def test_partial_page_resends_only_remaining_full_pool(monkeypatch) -> None:
    deliverables = tuple(f"deliverable_{index}" for index in range(7))
    first_done = deliverables[:3]
    second_done = deliverables[3:]
    pages = [
        {
            "modules": [_module(f"first_{index}") for index in range(3)],
            "assets": [],
            "audio": [],
            "acceptance_tests": [],
            "completed_deliverables": list(first_done),
            "complete": False,
            "next_cursor": "next",
        },
        {
            "modules": [_module(f"second_{index}") for index in range(4)],
            "assets": [],
            "audio": [],
            "acceptance_tests": [],
            "completed_deliverables": list(second_done),
            "complete": True,
            "next_cursor": "",
        },
    ]

    calls, parts = _run(monkeypatch, deliverables, pages)

    assert len(calls) == 2
    assert calls[0]["request"]["current_target_deliverables"] == list(deliverables)
    assert calls[1]["request"]["current_target_deliverables"] == list(second_done)
    assert calls[1]["request"]["remaining_deliverables"] == list(second_done)
    assert len(parts.modules) == 7
