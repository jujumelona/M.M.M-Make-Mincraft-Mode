from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai import small_model_task_capsule_contract as task_capsule
from minecraft_mod_ai.coder_mutation_authority_contract import install
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA

TASK_ID = "task_existing_source_edit"
JAVA_PATH = "src/main/java/example/ExistingFeature.java"
SYMBOL = "ExistingFeature"


def _module(status: str) -> SimpleNamespace:
    anchor = {
        "kind": "symbol",
        "locator": f"{JAVA_PATH}#{SYMBOL}",
        "status": status,
        "ownership": "exclusive",
        "module_id": "root",
        "source_set": "main",
    }
    task = {
        "task_id": TASK_ID,
        "task_sha256": "sha256:" + "b" * 64,
        "owned_anchors": [anchor],
        "required_gates": ["source_static_validation"],
        "production_bindings": [
            {
                "task_ref": TASK_ID,
                "reuse_action": "fresh",
                "owned_anchors": [anchor],
            }
        ],
    }
    return SimpleNamespace(
        module_id=TASK_ID,
        kind="custom_java",
        config={"evidence_task": task},
        depends_on=(),
        required_gates=("source_static_validation",),
    )


def _tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "tool",
            "parameters": json.loads(json.dumps(SOURCE_EDIT_SCHEMA)),
        },
    }


def _payload(status: str) -> dict:
    module = _module(status)
    return {
        "phase": "implement_module",
        "module": {
            "module_id": module.module_id,
            "kind": module.kind,
            "evidence_task": module.config["evidence_task"],
        },
    }


def setup_module() -> None:
    install(task_capsule, tool_loop)


def test_existing_primary_is_writable_but_not_creatable() -> None:
    capsule = task_capsule.compile_task_capsule(_module("existing"))
    assert capsule is not None
    assert capsule.writable_paths == (JAVA_PATH,)
    assert capsule.creatable_paths == ()
    assert capsule.anchor_for_path(JAVA_PATH).status == "existing"


def test_existing_primary_accepts_semantic_modification() -> None:
    capsule = task_capsule.compile_task_capsule(_module("existing"))
    assert capsule is not None
    bound = task_capsule.bind_source_edit_arguments(
        {
            "operation": "replace_exact",
            "path": JAVA_PATH,
            "old": "old",
            "new": "new",
        },
        capsule,
    )
    assert bound["path"] == JAVA_PATH
    assert bound["operation"] == "replace_exact"


def test_existing_primary_rejects_creation() -> None:
    capsule = task_capsule.compile_task_capsule(_module("existing"))
    assert capsule is not None
    with pytest.raises(task_capsule.TaskCapsuleContractError, match="CREATE_NOT_RESERVED"):
        task_capsule.bind_source_edit_arguments(
            {
                "operation": "create_java_type",
                "path": JAVA_PATH,
                "package_name": "example",
                "declaration": f"public final class {SYMBOL}",
            },
            capsule,
        )


def test_reserved_primary_remains_creatable() -> None:
    capsule = task_capsule.compile_task_capsule(_module("host_reserved"))
    assert capsule is not None
    assert capsule.creatable_paths == (JAVA_PATH,)
    bound = task_capsule.bind_source_edit_arguments(
        {
            "operation": "create_java_type",
            "path": JAVA_PATH,
            "package_name": "example",
            "declaration": f"public final class {SYMBOL}",
        },
        capsule,
    )
    assert bound["path"] == JAVA_PATH


def test_evidence_task_delete_is_always_rejected() -> None:
    capsule = task_capsule.compile_task_capsule(_module("existing"))
    assert capsule is not None
    with pytest.raises(task_capsule.TaskCapsuleContractError, match="DELETE_FORBIDDEN"):
        task_capsule.bind_source_edit_arguments(
            {"operation": "delete_file", "path": JAVA_PATH},
            capsule,
        )


def test_model_schema_hides_create_and_delete_when_only_existing_targets_exist() -> None:
    capsule = task_capsule.compile_task_capsule(_module("existing"))
    assert capsule is not None
    narrowed = task_capsule.narrow_source_edit_schema(_tool_schema(), capsule)
    enum = narrowed["function"]["parameters"]["properties"]["operation"]["enum"]
    assert "create" not in enum
    assert "create_file" not in enum
    assert "create_java_type" not in enum
    assert "delete" not in enum
    assert "delete_file" not in enum
    assert "replace_exact" in enum
    assert "insert_java_member" in enum


def test_fresh_reuse_action_does_not_turn_existing_anchor_into_new_file() -> None:
    context = tool_loop._fresh_owned_symbol_context(_payload("existing"))
    assert context is not None
    assert context.target_path == JAVA_PATH
    assert context.target_symbol == SYMBOL
    assert context.is_new_file is False
    assert context.localization_stage == tool_loop.LocalizationStage.NEED_BODY
    assert context.evidence_source == "evidence_existing_owned_anchor"


def test_host_reserved_anchor_is_still_a_new_file_target() -> None:
    context = tool_loop._fresh_owned_symbol_context(_payload("host_reserved"))
    assert context is not None
    assert context.target_path == JAVA_PATH
    assert context.is_new_file is True
    assert context.localization_stage == tool_loop.LocalizationStage.READY
    assert context.evidence_source == "evidence_host_reserved_owned_anchor"


def test_unknown_primary_status_still_fails_closed() -> None:
    with pytest.raises(task_capsule.TaskCapsuleContractError, match="PRIMARY_NOT_RESERVED"):
        task_capsule.compile_task_capsule(_module("planner_guess"))
