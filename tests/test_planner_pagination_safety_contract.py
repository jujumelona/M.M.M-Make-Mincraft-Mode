from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import complete_planner as planner_module
from minecraft_mod_ai.planner_pagination_safety_contract import install
from minecraft_mod_ai.spec import SpecValidationError


def _planner():
    install(planner_module)
    return planner_module.CompleteGameDesignPlanner(SimpleNamespace())


def _batch_page(*, module_id: str, completed: list[str], complete: bool, cursor: str):
    return {
        "modules": [
            {
                "module_id": module_id,
                "kind": "custom_java",
                "config": {},
                "depends_on": [],
                "required_gates": [],
            }
        ],
        "assets": [],
        "audio": [],
        "acceptance_tests": [f"verify {module_id}"],
        "completed_deliverables": completed,
        "complete": complete,
        "next_cursor": cursor,
    }


def test_production_batch_requires_monotonic_deliverable_progress(monkeypatch) -> None:
    planner = _planner()
    monkeypatch.setattr(
        planner_module,
        "_generate_json_page_with_repair",
        lambda *args, **kwargs: _batch_page(
            module_id="unrelated_output",
            completed=[],
            complete=False,
            cursor="c1",
        ),
    )
    parts = planner_module._ProductionParts([], [], [], [])
    with pytest.raises(SpecValidationError, match="made no verified progress"):
        planner._expand_one_production_batch(
            batch=planner_module._ProductionBatch(
                "core", "scope", (), ("first", "second"), ()
            ),
            parts=parts,
            module_catalog=planner_module._ModuleCatalog(),
            asset_catalog=planner_module._ModuleCatalog(),
            audio_catalog=planner_module._ModuleCatalog(),
            test_catalog=set(),
            dependency_exports={},
            planning_context={},
            planning_receipt={},
            media_paths=(),
        )
    assert parts.modules == []


def test_production_batch_advances_cursor_and_finishes(monkeypatch) -> None:
    planner = _planner()
    seen_requests: list[dict] = []
    pages = iter(
        [
            _batch_page(
                module_id="first_module",
                completed=["first"],
                complete=False,
                cursor="c1",
            ),
            _batch_page(
                module_id="second_module",
                completed=["second"],
                complete=True,
                cursor="",
            ),
        ]
    )

    def generate(*args, **kwargs):
        seen_requests.append(dict(kwargs["request"]))
        return next(pages)

    monkeypatch.setattr(planner_module, "_generate_json_page_with_repair", generate)
    parts = planner_module._ProductionParts([], [], [], [])
    planner._expand_one_production_batch(
        batch=planner_module._ProductionBatch(
            "core", "scope", (), ("first", "second"), ()
        ),
        parts=parts,
        module_catalog=planner_module._ModuleCatalog(),
        asset_catalog=planner_module._ModuleCatalog(),
        audio_catalog=planner_module._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={},
        planning_receipt={},
        media_paths=(),
    )
    assert [request["cursor"] for request in seen_requests] == ["", "c1"]
    assert [module.module_id for module in parts.modules] == [
        "first_module",
        "second_module",
    ]


def test_outline_repeated_cursor_fails_closed(monkeypatch) -> None:
    planner = _planner()
    first_page = {
        "production_batches": [
            {
                "batch_id": "first",
                "scope": "first scope",
                "depends_on_batches": [],
                "deliverables": ["one"],
                "exports": [],
            }
        ],
        "complete": False,
        "next_cursor": "c1",
    }
    repeated = {
        "production_batches": [
            {
                "batch_id": "second",
                "scope": "second scope",
                "depends_on_batches": [],
                "deliverables": ["two"],
                "exports": [],
            }
        ],
        "complete": False,
        "next_cursor": "c1",
    }
    monkeypatch.setattr(
        planner_module,
        "_generate_json_page_with_repair",
        lambda *args, **kwargs: repeated,
    )
    with pytest.raises(SpecValidationError, match="did not advance"):
        planner._collect_one_request_page_outline(
            first_page=first_page,
            base_request={},
            page_index=0,
            page_count=1,
        )
