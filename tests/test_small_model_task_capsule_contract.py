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
        "target_cell": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "yarn",
            "java_version": "21",
        },
        "owned_anchors": [
            main_anchor,
            test_anchor,
            {
                "kind": "registry",
                "locator": "registry:generated_mod:resource_gathering",
                "status": "host_reserved",
            },
        ],
        "implementation_obligations": [
            "Implement the approved resource gathering behavior in the owned source targets."
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


def test_custom_java_host_injects_target_compile_gate() -> None:
    module = _module()
    module.required_gates = ("source_static_validation",)

    capsule = compile_task_capsule(module)

    assert capsule is not None
    assert capsule.required_gates == ("source_static_validation", "target_compile")


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


def test_fresh_host_reserved_java_target_separates_write_and_api_evidence_authority() -> None:
    capsule = compile_task_capsule(_module(reuse_action="fresh"))
    assert capsule is not None
    messages = [
        {"role": "system", "content": "Implement the approved task."},
        {
            "role": "developer",
            "content": json.dumps(capsule.to_host_authority_payload()),
        },
    ]
    state = tool_loop.HostRunState()
    assert tool_loop.is_mutation_ready(messages, state) is True
    assert tool_loop._host_target_execution_authority(state) is True
    assert tool_loop._target_evidence_ready(
        state,
        require_rag=False,
        fresh_java_target=True,
    ) is True
    assert tool_loop._target_evidence_ready(
        state,
        require_rag=True,
        fresh_java_target=True,
        compile_backed_java=True,
    ) is True

    assert tool_loop._requires_rag_evidence(
        role="coder",
        host_grounded=False,
        router_requires_fresh_evidence=False,
        implementation_requires_mutation=True,
        initial_execution_authority=True,
    ) is False
    assert tool_loop._requires_rag_evidence(
        role="coder",
        host_grounded=False,
        router_requires_fresh_evidence=True,
        implementation_requires_mutation=True,
        initial_execution_authority=True,
    ) is True
    assert tool_loop._requires_rag_evidence(
        role="coder",
        host_grounded=False,
        router_requires_fresh_evidence=False,
        implementation_requires_mutation=True,
        initial_execution_authority=False,
    ) is True
    assert tool_loop._requires_rag_evidence(
        role="coder",
        host_grounded=False,
        router_requires_fresh_evidence=True,
        implementation_requires_mutation=False,
        initial_execution_authority=False,
    ) is True


def test_fresh_java_edit_schema_requires_one_complete_create_before_compile() -> None:
    context = tool_loop.TargetMutationContext(
        target_path=JAVA_PATH,
        target_symbol=SYMBOL,
        is_new_file=True,
        evidence_source="host_owned_fresh_target",
        writable_paths=(JAVA_PATH,),
        creatable_paths=(JAVA_PATH,),
        target_pinned=True,
    )
    narrowed = tool_loop._source_edit_schema_for_context(_tool_schema(), context)
    operations = narrowed["function"]["parameters"]["properties"]["operation"]["enum"]

    assert operations == ["create_file", "create"]


def test_materialized_java_edit_schema_allows_atomic_same_path_rewrite() -> None:
    context = tool_loop.TargetMutationContext(
        target_path=JAVA_PATH,
        target_symbol=SYMBOL,
        source_body="package generated.generated_mod; public final class X {}\n",
        is_new_file=False,
        evidence_source="mutation_receipt",
        writable_paths=(JAVA_PATH,),
        creatable_paths=(),
        target_pinned=True,
    )
    narrowed = tool_loop._source_edit_schema_for_context(_tool_schema(), context)
    operations = {
        value.casefold()
        for value in narrowed["function"]["parameters"]["properties"]["operation"]["enum"]
    }

    assert {"create_file", "create", "replace_exact"} <= operations
    assert "create_java_type" not in operations
    assert "SHA-bound replace" in narrowed["function"]["description"]


def test_fresh_java_optional_observe_frontier_stays_local() -> None:
    schemas = {
        name: {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }
        for name in (
            "search_code_rag",
            "java_workspace_symbols",
            "external_mcp_call",
            "inspect_modrinth_project",
        )
    }
    context = tool_loop.TargetMutationContext(
        target_path=JAVA_PATH,
        target_symbol=SYMBOL,
        is_new_file=True,
        evidence_source="evidence_fresh_owned_anchor",
        writable_paths=(JAVA_PATH,),
        creatable_paths=(JAVA_PATH,),
        target_pinned=True,
    )

    assert tool_loop._fresh_observe_names(
        schemas,
        set(),
        context,
        semantic_retrieval_choice=True,
    ) == ["search_code_rag", "java_workspace_symbols"]


def test_nested_current_project_java_hit_authorizes_fresh_java() -> None:
    value = {
        "structured_content": {
            "schema_version": "mmm/code-rag-result-v1",
            "hits": [
                {
                    "path": "src/main/java/dev/mmm/debugfixture/MmmDebugFixtureMod.java",
                    "text": (
                        "package dev.mmm.debugfixture;\n"
                        "import net.fabricmc.api.ModInitializer;\n"
                        "public final class MmmDebugFixtureMod implements ModInitializer {}\n"
                    ),
                }
            ],
            "receipt": {"status": "FOUND", "result_count": 1},
        }
    }

    assert tool_loop._authoritative_java_evidence(value) is True


def test_metadata_only_project_rag_does_not_authorize_fresh_java() -> None:
    metadata_only = {
        "schema_version": "mmm/rag-result-v1",
        "query": "register item",
        "minecraft_version": "26.2",
        "sources": [
            {
                "source_id": "fabric-project-creation",
                "title": "Fabric Documentation - Creating a Project",
                "url": "https://docs.fabricmc.net/develop/getting-started/creating-a-project",
                "authority": "Fabric official documentation",
                "version_scope": "Version-selected Fabric documentation",
            }
        ],
    }

    assert tool_loop._authoritative_java_evidence(metadata_only) is False

def test_external_mcp_retrieval_signatures_are_capability_specific() -> None:
    assert tool_loop.retrieval_query_signature(
        "external_mcp_call", {"capability": "source_search"}
    ) != tool_loop.retrieval_query_signature(
        "external_mcp_call", {"capability": "official_mod_docs"}
    )
    assert tool_loop.retrieval_source_key(
        "external_mcp_schema", {"capability": "source_search"}
    ) == "external_mcp_schema:source_search"
    assert tool_loop.retrieval_source_key(
        "external_mcp_call", {"capability": "source_search"}
    ) == "external_mcp_call:source_search"


def test_compact_coder_contract_drops_planner_provenance_blob() -> None:
    module = _module()
    original_task = module.config["evidence_task"]
    compact = compact_task_local_module_contract(module)
    compact_task = compact["evidence_task"]
    assert "request_context" not in compact_task
    assert compact_task["owned_anchors"] == original_task["owned_anchors"]
    assert compact_task["production_bindings"] == original_task["production_bindings"]
    assert compact_task["acceptance"] == original_task["acceptance"]
    assert compact_task["coder_execution_contract"]["schema_version"] == (
        "mmm/coder-execution-contract"
    )

    original_bytes = len(json.dumps(original_task).encode("utf-8"))
    compact_bytes = len(json.dumps(compact_task).encode("utf-8"))
    assert compact_bytes < original_bytes // 8
