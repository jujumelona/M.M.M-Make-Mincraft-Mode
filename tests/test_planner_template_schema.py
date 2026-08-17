from __future__ import annotations

import json

from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _host_batches
from minecraft_mod_ai.planner_template_schema import (
    ASSET_KEYS,
    MODULE_KEYS,
    TOP_LEVEL_KEYS,
    build_batch_skeleton,
    merge_model_output_into_skeleton,
)


def _skeleton() -> dict[str, object]:
    return build_batch_skeleton(
        batch_id="core_runtime",
        scope="Implement the core runtime.",
        deliverables=("runtime_done",),
        exports=("core_runtime_api",),
        depends_on_batches=("existing_module",),
        known_module_ids=("existing_module",),
    )


def test_host_owns_complete_page_shape() -> None:
    page = _skeleton()
    assert set(page) == set(TOP_LEVEL_KEYS)
    assert set(page["modules"][0]) == set(MODULE_KEYS)
    assert page["complete"] is True
    assert page["next_cursor"] == ""


def test_unknown_model_fields_are_discarded() -> None:
    page = merge_model_output_into_skeleton(
        skeleton=_skeleton(),
        model_output={
            "modules": [
                {
                    "module_id": "core_runtime_api",
                    "kind": "custom_java",
                    "config": {"feature": "runtime"},
                    "depends_on": ["existing_module", "invented_module"],
                    "required_gates": ["runtime"],
                    "legacy_audio": {"kind": "music"},
                }
            ],
            "assets": [],
            "acceptance_tests": ["test_runtime"],
            "completed_deliverables": ["runtime_done"],
            "complete": True,
            "next_cursor": "",
            "module_batches": [{"obsolete": True}],
            "audio": [{"obsolete": True}],
        },
        valid_module_catalog={"existing_module", "core_runtime_api"},
    )
    assert set(page) == set(TOP_LEVEL_KEYS)
    assert set(page["modules"][0]) == set(MODULE_KEYS)
    assert page["modules"][0]["depends_on"] == ["existing_module"]
    assert "legacy_audio" not in page["modules"][0]
    assert "module_batches" not in page
    assert "audio" not in page


def test_assets_are_closed_and_path_safe() -> None:
    page = merge_model_output_into_skeleton(
        skeleton=_skeleton(),
        model_output={
            "assets": [
                {
                    "asset_id": "runtime_icon",
                    "kind": "icon",
                    "prompt": "Minecraft runtime icon",
                    "target_path": "assets/mmm/textures/gui/runtime.png",
                    "width": 32,
                    "height": 32,
                    "unknown": "discard me",
                },
                {
                    "asset_id": "unsafe",
                    "kind": "icon",
                    "prompt": "unsafe",
                    "target_path": "../outside.png",
                },
            ]
        },
        valid_module_catalog={"existing_module", "core_runtime_api"},
    )
    assert len(page["assets"]) == 1
    assert set(page["assets"][0]) == set(ASSET_KEYS)
    assert page["assets"][0]["target_path"] == "assets/mmm/textures/gui/runtime.png"


def test_invalid_module_kind_falls_back_without_new_contract_layer() -> None:
    page = merge_model_output_into_skeleton(
        skeleton=_skeleton(),
        model_output={
            "modules": [
                {
                    "module_id": "core_runtime_api",
                    "kind": "music_generator",
                    "config": {},
                    "depends_on": [],
                    "required_gates": [],
                }
            ]
        },
        valid_module_catalog={"core_runtime_api"},
    )
    assert page["modules"][0]["kind"] == "custom_java"


def test_multiple_design_modules_use_one_host_template_fill() -> None:
    class Router:
        def __init__(self) -> None:
            self.calls = 0

        def generate_text(self, _role, messages, **_kwargs):
            self.calls += 1
            request = json.loads(messages[-1]["content"])
            return json.dumps(request["template_skeleton"])

    design = {
        "modules": [
            {"plugin_id": "combat", "reason": "combat behavior"},
            {"plugin_id": "economy", "reason": "economy behavior"},
            {"plugin_id": "quests", "reason": "quest behavior"},
        ]
    }
    batches = _host_batches("Build combat, economy, and quests", design)
    router = Router()
    modules, _assets, _tests = CompleteGameDesignPlanner(router)._expand_batches(
        batches,
        prompt="Build combat, economy, and quests",
        game_design=design,
    )

    assert len(batches) == 1
    assert batches[0].exports == ("combat", "economy", "quests")
    assert router.calls == 1
    assert tuple(module.module_id for module in modules) == (
        "combat",
        "economy",
        "quests",
    )
