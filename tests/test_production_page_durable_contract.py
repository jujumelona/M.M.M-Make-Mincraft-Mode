from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai import production_page_durable_contract as durable
from minecraft_mod_ai.planner_production_page_contract import install


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
        if len(self.calls) == 1:
            config = {
                "integration_type": "mmm_local_ai_sidecar",
                "capabilities": ["ai_inference"],
                "authentication": "invalid",
            }
        else:
            config = {
                "integration_type": "mmm_local_ai_sidecar",
                "capabilities": ["ai_inference"],
                "authentication": "none",
            }
        return json.dumps(
            {
                "target_fingerprint": request["target_fingerprint"],
                "set_fields": {"config": config},
                "delete_fields": [],
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
                "set_fields": {
                    "config": {
                        "integration_type": "mmm_local_ai_sidecar",
                        "capabilities": ["ai_inference"],
                        "authentication": "invalid",
                    }
                },
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
    install(complete_planner)

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


@pytest.mark.parametrize(
    "raw_kind,expected_kind",
    [
        ("", "custom_java"),
        ("config", "custom_java"),
        ("gradle", "custom_java"),
        ("network", "networking"),
        ("future_unknown_kind", "custom_java"),
    ],
)
def test_module_kind_parser_mismatches_are_normalized_without_llm(
    monkeypatch,
    tmp_path,
    raw_kind,
    expected_kind,
) -> None:
    router = _NeverRouter()
    page = {
        "modules": [_module("mk_gradle_config", kind=raw_kind)],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["platform module exists"],
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

    assert router.calls == []
    assert [item.module_id for item in parts.modules] == ["mk_gradle_config"]
    assert parts.modules[0].kind == expected_kind


def test_module_structural_fields_are_canonicalized_without_llm(monkeypatch, tmp_path) -> None:
    router = _NeverRouter()
    module = _module("self_module", kind="custom_java")
    module.update(
        {
            "config": {"implementation": "fabric"},
            "depends_on": ["self_module", "Bad-ID", "Bad-ID", ""],
            "required_gates": ["", " Gradle ", "Gradle"],
        }
    )
    page = {
        "modules": [module],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["module exists"],
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

    assert router.calls == []
    assert parts.modules[0].config["implementation"] == "custom"
    assert parts.modules[0].depends_on == ("bad_id",)
    assert parts.modules[0].required_gates == ("Gradle",)


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
                "kind": "sprite",
                "description": "Core item texture",
                "target_path": "../escape.png",
                "width": 0,
                "height": "not-an-int",
                "implements_deliverables": ["d1"],
            },
            {
                "kind": "unknown_texture_kind",
                "description": "Block texture",
                "target_path": "assets/mod/textures/block/second.png",
                "implements_deliverables": ["d2"],
            },
        ],
        "audio": [
            {
                "kind": "voice",
                "duration_seconds": "nan",
                "frequency_hz": 0,
                "volume": 99,
                "loop": "false",
                "subtitle_en": 123,
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
    assert [asset.kind for asset in parts.assets] == ["item", "block"]
    assert parts.assets[0].target_path == "assets/mod/textures/d1.png"
    assert (parts.assets[0].width, parts.assets[0].height) == (1, 16)
    assert [sound.sound_id for sound in parts.audio] == ["d2"]
    assert parts.audio[0].kind == "effect"
    assert parts.audio[0].duration_seconds == 1.0
    assert parts.audio[0].frequency_hz == 1.0
    assert parts.audio[0].volume == 4.0
    assert parts.audio[0].loop is False
    assert parts.audio[0].subtitle_en == "123"


def test_child_repair_schema_uses_parser_field_types_and_enums() -> None:
    install(complete_planner)

    module_schema = durable._patch_schema(
        fields=sorted(durable._SPECS["module"]["fields"]),
        replacement=False,
    )
    module_fields = module_schema["properties"]["set_fields"]["properties"]
    assert "custom_java" in module_fields["kind"]["enum"]
    assert module_fields["config"]["type"] == "object"
    assert module_fields["depends_on"]["items"]["minLength"] == 1
    assert set(module_schema["properties"]["delete_fields"]["items"]["enum"]) == set(
        durable._SPECS["module"]["fields"]
    )

    asset_schema = durable._patch_schema(
        fields=sorted(durable._SPECS["asset"]["fields"]),
        replacement=False,
    )
    asset_fields = asset_schema["properties"]["set_fields"]["properties"]
    assert asset_fields["width"]["type"] == "integer"
    assert "item" in asset_fields["kind"]["enum"]

    audio_schema = durable._patch_schema(
        fields=sorted(durable._SPECS["audio"]["fields"]),
        replacement=False,
    )
    audio_fields = audio_schema["properties"]["set_fields"]["properties"]
    assert audio_fields["duration_seconds"]["type"] == "number"
    assert audio_fields["loop"]["type"] == "boolean"
    assert "effect" in audio_fields["kind"]["enum"]


def _sidecar_module(authentication: str) -> dict[str, object]:
    value = _module("semantic_sidecar", kind="integration")
    value["config"] = {
        "integration_type": "mmm_local_ai_sidecar",
        "capabilities": ["ai_inference"],
        "authentication": authentication,
    }
    return value


def test_semantic_validation_keeps_field_patching_while_state_changes(
    monkeypatch,
    tmp_path,
) -> None:
    router = _EscalatingRouter()
    page = {
        "modules": [_sidecar_module("broken")],
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
        "field_patch",
    ]
    assert [item.module_id for item in parts.modules] == ["semantic_sidecar"]
    assert parts.modules[0].config["authentication"] == "none"


def test_repeated_invalid_model_output_stops_exact_cycle(
    monkeypatch,
    tmp_path,
) -> None:
    router = _StuckRouter()
    page = {
        "modules": [_sidecar_module("invalid")],
        "assets": [],
        "audio": [],
        "acceptance_tests": ["stuck item exists"],
        "completed_deliverables": ["d1"],
        "complete": True,
        "next_cursor": "",
    }

    with pytest.raises(
        complete_planner.SpecValidationError,
        match="repeated_(validation_state|model_output)",
    ):
        _run_batch(
            monkeypatch,
            tmp_path,
            page=page,
            deliverables=("d1",),
            router=router,
        )

    assert len(router.calls) == 1
