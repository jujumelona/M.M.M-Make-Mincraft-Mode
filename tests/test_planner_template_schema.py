from __future__ import annotations

from minecraft_mod_ai.implementation_template_contract import SCHEMA as CODER_SCHEMA
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


def _evidence_skeleton() -> dict[str, object]:
    task = {
        "task_id": "core_runtime_api",
        "task_sha256": "sha256:" + "a" * 64,
        "semantic_outcome": "The runtime behavior is observable in Minecraft.",
        "requirement_refs": ["req_runtime"],
        "depends_on": [],
        "consumes": [],
        "provides": ["runtime_done"],
        "implementation_capabilities": ["runtime.behavior"],
        "required_gates": ["source_static_validation", "target_compile"],
        "public_acceptance": ["The runtime behavior is observable."],
        "owned_anchors": [
            {
                "kind": "symbol",
                "locator": "src/main/java/demo/CoreRuntime.java#CoreRuntime",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            }
        ],
    }
    return build_batch_skeleton(
        batch_id="core_runtime",
        scope="Implement the core runtime.",
        deliverables=("runtime_done",),
        exports=("core_runtime_api",),
        host_module_contracts={
            "core_runtime_api": {
                **task,
                "evidence_plan_sha256": "sha256:" + "b" * 64,
                "evidence_task": task,
            }
        },
        acceptance_tests=("The runtime behavior is observable.",),
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


def test_evidence_page_contains_complete_host_coder_contract() -> None:
    page = _evidence_skeleton()
    module = page["modules"][0]
    config = module["config"]
    contract = config["coder_execution_contract"]

    assert contract["schema_version"] == CODER_SCHEMA
    assert contract["task_ref"] == "core_runtime_api"
    assert contract["targets"] == [
        {
            "kind": "symbol",
            "locator": "src/main/java/demo/CoreRuntime.java#CoreRuntime",
            "path": "src/main/java/demo/CoreRuntime.java",
            "symbol": "CoreRuntime",
            "operation": "create_or_modify",
            "module_id": ":",
            "source_set": "main",
        }
    ]
    assert contract["protected_boundaries"]["writable_paths"] == [
        "src/main/java/demo/CoreRuntime.java"
    ]
    assert module["required_gates"] == ["source_static_validation", "target_compile"]


def test_model_cannot_widen_evidence_owned_contract_or_acceptance() -> None:
    skeleton = _evidence_skeleton()
    original_config = skeleton["modules"][0]["config"]
    page = merge_model_output_into_skeleton(
        skeleton=skeleton,
        model_output={
            "modules": [
                {
                    "module_id": "core_runtime_api",
                    "kind": "boss",
                    "config": {
                        "evidence_task": {"task_id": "invented"},
                        "coder_execution_contract": {"targets": [{"path": "../escape"}]},
                        "implementation_notes": "non-authoritative note",
                    },
                    "depends_on": ["invented_module"],
                    "required_gates": ["invented_gate"],
                }
            ],
            "assets": [
                {
                    "asset_id": "invented",
                    "kind": "icon",
                    "prompt": "invented",
                    "target_path": "assets/invented.png",
                }
            ],
            "acceptance_tests": ["invented acceptance"],
            "completed_deliverables": ["invented deliverable"],
        },
        valid_module_catalog={"core_runtime_api"},
    )
    module = page["modules"][0]
    config = module["config"]

    assert module["kind"] == "custom_java"
    assert module["depends_on"] == []
    assert module["required_gates"] == ["source_static_validation", "target_compile"]
    assert config["evidence_task"] == original_config["evidence_task"]
    assert config["coder_execution_contract"] == original_config["coder_execution_contract"]
    assert config["model_fill"] == {"implementation_notes": "non-authoritative note"}
    assert page["assets"] == []
    assert page["acceptance_tests"] == ["The runtime behavior is observable."]
    assert page["completed_deliverables"] == ["runtime_done"]
