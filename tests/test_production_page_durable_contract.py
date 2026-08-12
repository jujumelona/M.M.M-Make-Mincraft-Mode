from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import complete_planner, work_graph
from minecraft_mod_ai.execution_efficiency_contract import install


class _NeverRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role: str,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ) -> str:
        self.calls.append({"role": role, "messages": messages})
        raise AssertionError("deterministic planner repair should not call the LLM")


class _EscalatingRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role: str,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ) -> str:
        request = json.loads(messages[-1]["content"])
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "request": request,
                "response_format": response_format,
            }
        )
        if request["repair_mode"] == "field_patch":
            return json.dumps(
                {
                    "target_fingerprint": request["target_fingerprint"],
                    "set_fields": {"kind": "still_not_a_real_kind"},
                    "delete_fields": [],
                }
            )
        return json.dumps(
            {
                "target_fingerprint": request["target_fingerprint"],
                "replacement": {
                    "module_id": "semantic_fixed",
                    "kind": "item",
                    "config": {},
                    "depends_on": [],
                    "required_gates": [],
                    "implements_deliverables": ["d1"],
                },
            }
        )


class _StuckRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role: str,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ) -> str:
        request = json.loads(messages[-1]["content"])
        self.calls.append({"role": role, "request": request})
        return json.dumps(
            {
                "target_fingerprint": request["target_fingerprint"],
                "set_fields": {"kind": "not_a_real_kind"},
                "delete_fields": [],
            }
        )


def _module(module_id: str, *, config: bool = True, kind: str = "item") -> dict[str, object]:
    value: dict[str, object] = {
        "module_id": module_id,
        "kind": kind,
        "depends_on": [],
        "required_gates": [],
        "implements_deliverables": [module_id],
    }
    if config:
        value["config"] = {}
    return value


def _run_batch(
    monkeypatch,
    tmp_path,
    *,
    page: dict[str, object],
    deliverables: tuple[str, ...],
    router,
):
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    install(complete_planner_module=complete_planner, work_graph_module=work_graph)

    page_calls = {"count": 0}

    def page_generator(*args, **kwargs):
        page_calls["count"] += 1
        return page

    monkeypatch.setattr(
        complete_planner,
        "_generate_json_page_with_repair",
        page_generator,
    )

    planner = object.__new__(complete_planner.CompleteGameDesignPlanner)
    planner.router = router
    parts = complete_planner._ProductionParts([], [], [], [])
    planner._expand_one_production_batch(
        batch=complete_planner._ProductionBatch(
            batch_id="durable_items",
            scope="test batch",
            depends_on_batches=(),
            deliverables=deliverables,
            exports=(),
        ),
        parts=parts,
        module_catalog=complete_planner._ModuleCatalog(),
        asset_catalog=complete_planner._ModuleCatalog(),
        audio_catalog=complete_planner._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={},
        planning_receipt={},
        media_paths=(),
    )
    return parts, page_calls["count"]


def test_parser_normalizable_module_does_not_spend_llm_call(monkeypatch, tmp_path) -> None:
    router = _NeverRouter()
    page = {
        "modules": [
            _module("good"),
            _module("needs_config", config=False),
        ],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["good and needs_config exist"],
        "completed_deliverables": ["good", "needs_config"],
        "complete": True,
        "next_cursor": "",
    }

    parts, page_calls = _run_batch(
        monkeypatch,
        tmp_path,
        page=page,
        deliverables=("good", "needs_config"),
        router=router,
    )

    assert page_calls == 1
    assert router.calls == []
    assert [item.module_id for item in parts.modules] == ["good", "needs_config"]

    parts_replayed, replay_page_calls = _run_batch(
        monkeypatch,
        tmp_path,
        page=page,
        deliverables=("good", "needs_config"),
        router=router,
    )
    assert replay_page_calls == 0
    assert [item.module_id for item in parts_replayed.modules] == [
        "good",
        "needs_config",
    ]


def test_duplicate_ids_are_resolved_deterministically_without_llm(monkeypatch, tmp_path) -> None:
    router = _NeverRouter()
    page = {
        "modules": [_module("duplicate"), _module("duplicate")],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["both implementations exist"],
        "completed_deliverables": ["d1", "d2"],
        "complete": True,
        "next_cursor": "",
    }

    parts, _ = _run_batch(
        monkeypatch,
        tmp_path,
        page=page,
        deliverables=("d1", "d2"),
        router=router,
    )

    assert router.calls == []
    assert [item.module_id for item in parts.modules] == [
        "duplicate",
        "duplicate_2",
    ]


def test_asset_audio_parser_contract_mismatches_are_normalized_without_llm(
    monkeypatch,
    tmp_path,
) -> None:
    router = _NeverRouter()
    page = {
        "modules": [_module("core")],
        "assets": [
            {
                "kind": "texture",
                "description": "Core item texture",
                "implements_deliverables": ["d1"],
            },
            {
                "kind": "texture",
                "description": "Second item texture",
                "implements_deliverables": ["d2"],
            },
        ],
        "audio": [
            {
                "kind": "sfx",
                "duration_seconds": 1,
                "loop": "false",
                "implements_deliverables": ["d2"],
            }
        ],
        "acceptance_tests": ["assets and audio exist"],
        "completed_deliverables": ["d1", "d2"],
        "complete": True,
        "next_cursor": "",
    }

    parts, _ = _run_batch(
        monkeypatch,
        tmp_path,
        page=page,
        deliverables=("d1", "d2"),
        router=router,
    )

    assert router.calls == []
    assert [asset.asset_id for asset in parts.assets] == ["d1", "d2"]
    assert all(asset.kind == "item" for asset in parts.assets)
    assert [sound.sound_id for sound in parts.audio] == ["d2"]
    assert parts.audio[0].kind == "effect"
    assert parts.audio[0].loop is False


def test_semantic_validation_uses_field_patch_then_single_item_regeneration(
    monkeypatch,
    tmp_path,
) -> None:
    router = _EscalatingRouter()
    page = {
        "modules": [_module("semantic_bad", kind="not_a_real_kind")],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["semantic item exists"],
        "completed_deliverables": ["d1"],
        "complete": True,
        "next_cursor": "",
    }

    parts, _ = _run_batch(
        monkeypatch,
        tmp_path,
        page=page,
        deliverables=("d1",),
        router=router,
    )

    assert [call["request"]["repair_mode"] for call in router.calls] == [
        "field_patch",
        "replacement",
    ]
    assert [item.module_id for item in parts.modules] == ["semantic_fixed"]
    assert parts.modules[0].kind == "item"


def test_repeated_invalid_state_is_cut_off_after_two_distinct_repair_modes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_ITEM_REPAIR_ATTEMPTS", "4")
    router = _StuckRouter()
    page = {
        "modules": [_module("stuck", kind="not_a_real_kind")],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["stuck item exists"],
        "completed_deliverables": ["d1"],
        "complete": True,
        "next_cursor": "",
    }

    with pytest.raises(
        complete_planner.SpecValidationError,
        match="repeated_model_output",
    ):
        _run_batch(
            monkeypatch,
            tmp_path,
            page=page,
            deliverables=("d1",),
            router=router,
        )

    assert len(router.calls) == 2
