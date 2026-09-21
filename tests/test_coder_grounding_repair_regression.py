from __future__ import annotations

import hashlib
import json

from minecraft_mod_ai.grounding_policy import host_baseline_evidence_ready
from minecraft_mod_ai.host_grounding import _SCHEMA_VERSION, build_coder_grounding
from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    LoopPhase,
    TargetMutationContext,
    _filter_tools_for_phase,
)


def _grounding_message(observation_count: int) -> dict[str, str]:
    payload = {
        "host_grounding": {
            "schema_version": _SCHEMA_VERSION,
            "policy": {
                "resolved_before_first_coder_decode": True,
                "baseline_grounding_owned_by_host": True,
                "baseline_grounding_optional_for_model": False,
                "model_tool_choice_required_for_baseline": False,
            },
            "evidence_bindings": {
                "project_exact_rag": {
                    "receipt": {
                        "project_sha256": "a" * 64,
                        "observations_sha256": "b" * 64,
                        "observation_count": observation_count,
                    }
                }
            },
        }
    }
    return {"role": "user", "content": json.dumps(payload)}


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
                        "enum": ["create_file", "create_java_type", "replace_exact", "insert_after"],
                    },
                    "path": {"type": "string"},
                },
            },
        },
    }


def test_empty_project_observation_receipt_is_not_grounded() -> None:
    assert host_baseline_evidence_ready((_grounding_message(0),)) is False
    assert host_baseline_evidence_ready((_grounding_message(1),)) is True


def test_coder_grounding_uses_canonical_native_target_java() -> None:
    grounding = build_coder_grounding(
        module_kind="debug_token",
        source_observation_receipt={
            "schema_version": "receipt-v1",
            "project_sha256": "a" * 64,
            "query_sha256": "b" * 64,
            "observation_count": 1,
            "observations_sha256": "c" * 64,
        },
        research_context={"schema_version": "research-v1", "selected_fact_count": 0},
        minecraft_version="26.2",
        loader="fabric",
        mappings="",
    )

    target = grounding["target"]
    assert target["minecraft_version"] == "26.2"
    assert target["java"] == "25"
    assert target["naming_regime"] == "native_unobfuscated"
    assert target["mappings_applicable"] is False
    assert target["mappings"] == ""


def test_verifier_repair_guidance_contains_current_source_and_hash() -> None:
    source = "package dev.mmm;\npublic final class DebugToken {}\n"
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path="src/main/java/dev/mmm/DebugToken.java",
            source_body=source,
            is_new_file=False,
            writable_paths=("src/main/java/dev/mmm/DebugToken.java",),
            target_pinned=True,
        )
    )
    state.validation_status = "FAIL"
    state.latest_verifier_tool = "java_diagnostics"
    state.latest_verifier_errors = ({"message": "cannot find symbol"},)
    state.latest_verifier_fingerprint = "verifier-fingerprint"

    guidance = state.take_verifier_repair_guidance()

    assert guidance is not None
    assert "MMM_CORE_VERIFIER_REPAIR_V5" in guidance
    payload = json.loads(guidance.rsplit("\n", 1)[-1])
    assert payload["current_source"] == source
    assert (
        payload["current_source_sha256"]
        == hashlib.sha256(source.encode("utf-8")).hexdigest()
    )
    assert payload["target_is_new_file"] is False
    assert "same-path create_file" not in guidance
    assert "host binds operation=replace_exact" in guidance


def test_existing_target_schema_does_not_offer_create_operations() -> None:
    context = TargetMutationContext(
        target_path="src/main/java/dev/mmm/DebugToken.java",
        source_body="public final class DebugToken {}",
        is_new_file=False,
        writable_paths=("src/main/java/dev/mmm/DebugToken.java",),
        target_pinned=True,
    )

    selected = _filter_tools_for_phase(
        (_source_edit_tool(),),
        LoopPhase.ACT,
        "coder",
        mutation_context=context,
    )

    operation = selected[0]["function"]["parameters"]["properties"]["operation"]
    assert operation["enum"] == ["replace_exact", "insert_after"]


def test_fresh_java_observe_prefers_exact_code_before_workspace_symbols() -> None:
    context = TargetMutationContext(
        target_path="src/main/java/dev/mmm/DebugToken.java",
        is_new_file=True,
        writable_paths=("src/main/java/dev/mmm/DebugToken.java",),
        creatable_paths=("src/main/java/dev/mmm/DebugToken.java",),
        target_pinned=True,
    )
    tools = (
        {"type": "function", "function": {"name": "search_project_rag", "parameters": {}}},
        {"type": "function", "function": {"name": "java_workspace_symbols", "parameters": {}}},
        {"type": "function", "function": {"name": "search_code_rag", "parameters": {}}},
    )

    selected = _filter_tools_for_phase(
        tools,
        LoopPhase.OBSERVE,
        "coder",
        mutation_context=context,
        semantic_retrieval_choice=True,
    )

    assert selected[0]["function"]["name"] == "search_code_rag"
