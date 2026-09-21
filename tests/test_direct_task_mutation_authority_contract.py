from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    DirectTaskMutationAuthorityError,
    compile_direct_task_mutation_authority,
)

TASK_ID = "task_space_mode_resource_gathering_semant_8e2529cd15"
JAVA_PATH = (
    "src/main/java/generated/generated_mod/mmmplan/"
    "TaskSpaceModeResourceGatheringSemant8e2529cd15.java"
)
TEST_PATH = (
    "src/test/java/generated/generated_mod/mmmplan/"
    "TaskSpaceModeResourceGatheringSemant8e2529cd15Test.java"
)


def _module(*, include_binding: bool = True):
    primary = {
        "kind": "symbol",
        "locator": f"{JAVA_PATH}#TaskSpaceModeResourceGatheringSemant8e2529cd15",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    test = {
        "kind": "test",
        "locator": f"{TEST_PATH}#TaskSpaceModeResourceGatheringSemant8e2529cd15Test",
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    task = {
        "task_id": TASK_ID,
        "task_sha256": "sha256:" + "a" * 64,
        "owned_anchors": [
            primary,
            test,
            {
                "kind": "registry",
                "locator": "registry:generated_mod:resource_gathering",
                "status": "host_reserved",
            },
        ],
        "production_bindings": (
            [
                {
                    "task_ref": TASK_ID,
                    "reuse_action": "fresh",
                    "owned_anchors": [primary],
                }
            ]
            if include_binding
            else []
        ),
    }
    return SimpleNamespace(
        module_id=TASK_ID,
        kind="custom_java",
        config={"evidence_task": task},
    )


def test_direct_authority_preserves_plan_java_and_test_exact_set() -> None:
    authority = compile_direct_task_mutation_authority(_module())

    assert authority is not None
    assert authority.primary_path == JAVA_PATH
    assert authority.writable_paths == (JAVA_PATH, TEST_PATH)
    assert authority.creatable_paths == (JAVA_PATH, TEST_PATH)
    assert all("registry:" not in path for path in authority.writable_paths)

    writable, creatable = tool_loop._planir_owned_anchor_sets(authority.to_host_payload())
    assert writable == (JAVA_PATH, TEST_PATH)
    assert creatable == (JAVA_PATH, TEST_PATH)


def test_direct_authority_preempts_unrelated_manifest_observation() -> None:
    authority = compile_direct_task_mutation_authority(_module())
    assert authority is not None
    messages = [
        {"role": "system", "content": "Implement the approved task."},
        {"role": "developer", "content": json.dumps(authority.to_host_payload())},
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
    assert JAVA_PATH in state.mutation_context.creatable_paths
    assert TEST_PATH in state.mutation_context.creatable_paths
    assert (
        tool_loop._mutation_target_error(
            "apply_source_edit",
            {"operation": "create_file", "path": JAVA_PATH},
            state.mutation_context,
        )
        is None
    )
    manifest_error = tool_loop._mutation_target_error(
        "apply_source_edit",
        {"operation": "replace_exact", "path": "src/main/resources/fabric.mod.json"},
        state.mutation_context,
    )
    assert manifest_error is not None
    assert "MUTATION_TARGET_DRIFT" in manifest_error


def test_existing_exact_task_gets_direct_authority_without_create_permission() -> None:
    module = _module()
    task = dict(module.config["evidence_task"])
    anchors = []
    for anchor in task["owned_anchors"]:
        item = dict(anchor)
        if item.get("kind") in {"symbol", "test"}:
            item["status"] = "existing"
        anchors.append(item)
    task["owned_anchors"] = anchors
    binding = dict(task["production_bindings"][0])
    binding["owned_anchors"] = [
        dict(anchors[0]),
    ]
    task["production_bindings"] = [binding]
    module.config = {"evidence_task": task}

    authority = compile_direct_task_mutation_authority(module)

    assert authority is not None
    assert authority.primary_path == JAVA_PATH
    assert authority.writable_paths == (JAVA_PATH, TEST_PATH)
    assert authority.creatable_paths == ()
    assert authority.mutation_authority.mutation_error(
        JAVA_PATH,
        operation="replace_exact",
    ) is None
    create_error = authority.mutation_authority.mutation_error(
        "src/main/java/example/Other.java",
        operation="create_file",
    )
    assert create_error is not None
    assert create_error.startswith("MUTATION_TARGET_DRIFT:")


def test_fresh_host_reserved_task_cannot_silently_fall_back_without_binding() -> None:
    with pytest.raises(DirectTaskMutationAuthorityError, match="BINDING_MISSING"):
        compile_direct_task_mutation_authority(_module(include_binding=False))
