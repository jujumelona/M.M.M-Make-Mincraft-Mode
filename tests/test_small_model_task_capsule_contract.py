from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.model_adapters import ToolCall
from minecraft_mod_ai.small_model_task_capsule_contract import (
    TaskCapsuleContractError,
    _bind_tool_call,
    _tool_allowed_for_capsule,
    assert_installed,
    bind_source_edit_arguments,
    compact_task_local_module_contract,
    compile_task_capsule,
    narrow_source_edit_schema,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA

TASK_ID = "task_space_mode_resource_gathering_semant_8e2529cd15"
JAVA_PATH = (
    "src/main/java/generated/generated_mod/mmmplan/"
    "TaskSpaceModeResourceGatheringSemant8e2529cd15.java"
)
TEST_PATH = (
    "src/test/java/generated/generated_mod/mmmplan/"
    "TaskSpaceModeResourceGatheringSemant8e2529cd15Test.java"
)
SYMBOL = "TaskSpaceModeResourceGatheringSemant8e2529cd15"


def _module(*, binding: bool = True, reuse_action: str = "fresh"):
    main_anchor = {
        "kind": "symbol",
        "locator": f"{JAVA_PATH}#{SYMBOL}",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    test_anchor = {
        "kind": "test",
        "locator": f"{TEST_PATH}#{SYMBOL}Test",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "test",
    }
    task = {
        "task_id": TASK_ID,
        "task_sha256": "sha256:" + "a" * 64,
        "requirement_refs": ["req_resource_gathering"],
        "gap_refs": ["gap_resource_gathering"],
        "owned_anchors": [
            main_anchor,
            test_anchor,
            {
                "kind": "registry",
                "locator": "registry:generated_mod:resource_gathering",
                "status": "host_reserved",
            },
        ],
        "provides": ["capability:space_mode_resource_gathering"],
        "acceptance": ["resource gathering changes an observable player resource state"],
        "required_gates": ["source_static_validation"],
        "production_bindings": (
            [
                {
                    "task_ref": TASK_ID,
                    "reuse_action": reuse_action,
                    "owned_anchors": [main_anchor],
                }
            ]
            if binding
            else []
        ),
        "request_context": {
            "requested_prompt": "x" * 48000,
            "requirements": [{"statement": "y" * 12000}],
            "planner_provenance": "z" * 12000,
        },
    }
    return SimpleNamespace(
        module_id=TASK_ID,
        kind="custom_java",
        config={"evidence_task": task},
        depends_on=(),
        required_gates=("source_static_validation", "target_compile"),
    )


def _tool_schema(name: str = "apply_source_edit") -> dict:
    parameters = json.loads(json.dumps(SOURCE_EDIT_SCHEMA)) if name == "apply_source_edit" else {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "tool",
            "parameters": parameters,
        },
    }


def test_runtime_installs_task_capsule_as_final_contract() -> None:
    assert_installed()
    assert getattr(tool_loop.generate_with_tools, "_mmm_small_model_task_capsule", False)


def test_capsule_compiles_exact_planir_main_and_test_authority() -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None
    assert capsule.primary_path == JAVA_PATH
    assert capsule.primary_symbol == SYMBOL
    assert capsule.writable_paths == (JAVA_PATH, TEST_PATH)
    assert capsule.creatable_paths == (JAVA_PATH, TEST_PATH)
    assert capsule.test_paths == (TEST_PATH,)
    assert capsule.required_gates == ("source_static_validation", "target_compile")

    writable, creatable = tool_loop._planir_owned_anchor_sets(
        capsule.to_host_authority_payload()
    )
    assert writable == (JAVA_PATH, TEST_PATH)
    assert creatable == (JAVA_PATH, TEST_PATH)


def test_planir_authority_fails_before_coder_when_binding_is_missing() -> None:
    with pytest.raises(TaskCapsuleContractError, match="BINDING_MISSING"):
        compile_task_capsule(_module(binding=False))


def test_reuse_changes_ingredients_not_destination_authority() -> None:
    capsule = compile_task_capsule(_module(reuse_action="adapt"))
    assert capsule is not None
    assert capsule.reuse_action == "adapt"
    assert capsule.primary_path == JAVA_PATH
    assert capsule.writable_paths == (JAVA_PATH, TEST_PATH)
    payload = capsule.to_host_authority_payload()
    assert payload["reuse_action"] == "adapt"
    assert payload["module"]["config"]["evidence_task"]["production_bindings"][0]["reuse_action"] == "adapt"

    writable, creatable = tool_loop._planir_owned_anchor_sets(payload)
    assert writable == (JAVA_PATH, TEST_PATH)
    assert creatable == (JAVA_PATH, TEST_PATH)


def test_small_model_schema_exposes_only_exact_host_paths() -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None
    narrowed = narrow_source_edit_schema(_tool_schema(), capsule)
    properties = narrowed["function"]["parameters"]["properties"]
    assert properties["path"]["enum"] == [JAVA_PATH, TEST_PATH]
    assert "file" not in properties
    assert "target_path" not in properties
    assert "target_file" not in properties


def test_hallucinated_model_path_is_rejected_as_mutation_target_drift() -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None
    hallucinated = (
        "src/main/java/ai/minecraft/generated/space_odyssey_fabric_mod/"
        f"{SYMBOL}.java"
    )
    with pytest.raises(TaskCapsuleContractError, match="MUTATION_TARGET_DRIFT"):
        bind_source_edit_arguments(
            {"operation": "create_file", "path": hallucinated, "content": "class X {}"},
            capsule,
        )

    call = ToolCall(
        id="call-1",
        name="apply_source_edit",
        arguments={"operation": "create_file", "path": hallucinated, "content": "class X {}"},
        raw_arguments="{}",
    )
    with pytest.raises(TaskCapsuleContractError, match="MUTATION_TARGET_DRIFT"):
        _bind_tool_call(call, capsule)


def test_wrong_test_path_cannot_be_rebound_into_owned_test_anchor() -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None
    with pytest.raises(TaskCapsuleContractError, match="MUTATION_TARGET_DRIFT"):
        bind_source_edit_arguments(
            {
                "operation": "create_file",
                "path": f"src/test/java/wrong/package/{SYMBOL}Test.java",
                "content": "class WrongTest {}",
            },
            capsule,
        )


def test_omitted_path_can_bind_to_unambiguous_host_primary() -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None
    bound = bind_source_edit_arguments(
        {"operation": "create_file", "content": "class X {}"}, capsule
    )
    assert bound["path"] == JAVA_PATH


def test_fresh_task_does_not_expose_reuse_tool() -> None:
    capsule = compile_task_capsule(_module(reuse_action="fresh"))
    assert capsule is not None
    assert _tool_allowed_for_capsule(_tool_schema("read_reuse_source"), capsule) is False
    assert _tool_allowed_for_capsule(_tool_schema("apply_source_edit"), capsule) is True


def test_adapt_task_may_expose_reuse_tool() -> None:
    capsule = compile_task_capsule(_module(reuse_action="adapt"))
    assert capsule is not None
    assert _tool_allowed_for_capsule(_tool_schema("read_reuse_source"), capsule) is True


def test_fabric_manifest_observation_cannot_become_task_mutation_target() -> None:
    capsule = compile_task_capsule(_module())
    assert capsule is not None
    messages = [
        {"role": "system", "content": "Implement the approved task."},
        {
            "role": "developer",
            "content": json.dumps(capsule.to_host_authority_payload()),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "target_file": "src/main/resources/fabric.mod.json",
                    "source": '{"schemaVersion":1,"id":"generated_mod"}',
                }
            ),
        },
    ]
    state = tool_loop.HostRunState()
    assert tool_loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.target_path == JAVA_PATH
    assert state.mutation_context.target_pinned is True


def test_compact_coder_contract_drops_planner_provenance_blob() -> None:
    module = _module()
    original_task = module.config["evidence_task"]
    compact = compact_task_local_module_contract(module)
    compact_task = compact["evidence_task"]
    assert "request_context" not in compact_task
    assert compact_task["owned_anchors"] == original_task["owned_anchors"]
    assert compact_task["production_bindings"] == original_task["production_bindings"]
    assert compact_task["acceptance"] == original_task["acceptance"]

    original_bytes = len(json.dumps(original_task).encode("utf-8"))
    compact_bytes = len(json.dumps(compact_task).encode("utf-8"))
    assert compact_bytes < original_bytes // 8
