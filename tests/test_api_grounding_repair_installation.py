from __future__ import annotations

from minecraft_mod_ai.api_grounding_repair_installation import (
    _completion_boundary_error,
    authoritative_java_evidence,
    strict_usable_rag_result,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    LoopPhase,
    TargetMutationContext,
    _filter_tools_for_phase,
    _mutation_target_error,
)


def _fresh_context() -> TargetMutationContext:
    return TargetMutationContext(
        target_path="src/main/java/dev/mmm/debugfixture/DebugToken.java",
        is_new_file=True,
        writable_paths=("src/main/java/dev/mmm/debugfixture/DebugToken.java",),
        creatable_paths=("src/main/java/dev/mmm/debugfixture/DebugToken.java",),
        target_pinned=True,
        evidence_source="host_task_authority",
    )


def _source_edit_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one source mutation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "create_file",
                            "create_java_type",
                            "replace_exact",
                            "insert_before",
                            "insert_after",
                        ],
                    },
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }


def test_metadata_only_project_rag_from_failure_trace_is_not_usable() -> None:
    payload = {
        "_mmm_observation": {
            "sanitized": True,
            "truncated": False,
            "trust": "untrusted_data_only",
        },
        "parsed_text": None,
        "resources": [],
        "structured_content": {
            "query": "DebugToken.java src/main/java/dev/mmm/debugfixture",
            "schema_version": "mmm/rag-result-v2",
            "sources": [
                {
                    "authority": "official",
                    "record_sha256": "sha256:deadbeef",
                    "retrieval_policy": "primary",
                    "source_id": "catalog-entry",
                    "title": "metadata only",
                    "trust_tier": "primary",
                    "url": "https://example.invalid",
                    "verified_on": "2026-09-16",
                    "version_scope": "26.2",
                }
            ],
            "target": {
                "loader": "fabric",
                "mappings": "",
                "minecraft_version": "26.2",
            },
        },
        "text": [],
    }

    assert strict_usable_rag_result(payload) is False
    assert authoritative_java_evidence(payload) is False


def test_fresh_java_does_not_promote_generic_metadata_to_grounding() -> None:
    state = HostRunState(mutation_context=_fresh_context())
    metadata = {
        "schema_version": "mmm/rag-result-v2",
        "query": "debug token item",
        "sources": [{"source_id": "only-provenance"}],
        "target": {"minecraft_version": "26.2", "loader": "fabric"},
    }

    assert state.record_evidence(metadata, usable=True) is False
    assert state.has_fresh_evidence is False


def test_fresh_java_accepts_concrete_workspace_symbols() -> None:
    state = HostRunState(mutation_context=_fresh_context())
    evidence = {
        "schema_version": "mmm/java-symbols-v1",
        "project_root": ".",
        "query": "Item",
        "symbols": [
            {
                "name": "Item",
                "kind": 5,
                "location": {"uri": "file:///workspace/src/main/java/example/Items.java"},
            }
        ],
    }

    assert authoritative_java_evidence(evidence) is True
    assert state.record_evidence(evidence, usable=True) is True
    assert state.has_fresh_evidence is True


def test_fresh_java_accepts_contentful_code_rag() -> None:
    evidence = {
        "schema_version": "mmm/code-rag-result-v1",
        "query": "item registration",
        "hits": [
            {
                "source_path": "src/main/java/dev/mmm/ExistingItems.java",
                "text": "package dev.mmm; import net.minecraft.world.item.Item; final class ExistingItems {}",
            }
        ],
        "receipt": {"result_count": 1},
    }

    assert strict_usable_rag_result(evidence) is True
    assert authoritative_java_evidence(evidence) is True


def test_fresh_java_never_falls_back_to_target_neutral_project_rag_authority() -> None:
    context = _fresh_context()
    tools = (
        {"type": "function", "function": {"name": "search_project_rag", "parameters": {}}},
        {"type": "function", "function": {"name": "search_code_rag", "parameters": {}}},
        {"type": "function", "function": {"name": "external_mcp_call", "parameters": {}}},
        {"type": "function", "function": {"name": "java_workspace_symbols", "parameters": {}}},
    )

    selected = _filter_tools_for_phase(
        tools,
        LoopPhase.OBSERVE,
        "coder",
        mutation_context=context,
        attempted_sources={
            "search_code_rag",
            "external_mcp_call:source_search",
            "java_workspace_symbols",
        },
        semantic_retrieval_choice=True,
    )

    assert selected == ()


def test_verifier_repair_reuses_create_file_as_same_path_transactional_replace() -> None:
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    context = TargetMutationContext(
        target_path=target,
        source_body="package dev.mmm.debugfixture;\npublic final class DebugToken {}\n",
        is_new_file=False,
        evidence_source="mutation_receipt",
        writable_paths=(target,),
        target_pinned=True,
    )

    selected = _filter_tools_for_phase(
        (_source_edit_tool(),),
        LoopPhase.ACT,
        "coder",
        mutation_context=context,
    )
    operation = selected[0]["function"]["parameters"]["properties"]["operation"]

    assert "create_file" in operation["enum"]
    assert _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "create_file",
            "path": target,
            "content": "package dev.mmm.debugfixture;\npublic final class DebugToken { int fixed; }\n",
        },
        context,
    ) is None


def test_create_file_is_still_rejected_for_unrelated_existing_target() -> None:
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    context = TargetMutationContext(
        target_path=target,
        source_body="public final class DebugToken {}",
        is_new_file=False,
        evidence_source="host_exact_source",
        writable_paths=(target,),
        target_pinned=True,
    )

    error = _mutation_target_error(
        "apply_source_edit",
        {"operation": "create_file", "path": target, "content": "class DebugToken {}"},
        context,
    )
    assert error is not None
    assert error.startswith("MUTATION_TARGET_CREATION_CONFLICT")


def test_completion_boundary_detection_is_semantic_not_retry_count_based() -> None:
    class LlamaCompletionBoundaryError(RuntimeError):
        pass

    assert _completion_boundary_error(LlamaCompletionBoundaryError("token boundary")) is True
    assert _completion_boundary_error(RuntimeError("ordinary backend failure")) is False
