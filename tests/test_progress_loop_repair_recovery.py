from __future__ import annotations

from minecraft_mod_ai import progress_aware_tool_loop
from minecraft_mod_ai.mutation_failure_classification import (
    is_post_argument_semantic_failure_code as _is_post_argument_semantic_failure_code,
    is_recoverable_mutation_failure,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA
from minecraft_mod_ai.validation_diagnostic_contract import diagnostic_errors


def _apply_source_edit_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "edit",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }


def _repair_messages(path: str) -> list[dict[str, str]]:
    payload = (
        '{"diagnostics":[],"target_is_new_file":false,'
        f'"target_path":"{path}","paths_created_in_this_run":["{path}"]}}'
    )
    return [
        {
            "role": "system",
            "content": "MMM_CORE_VERIFIER_REPAIR_V1\nrepair existing target\n" + payload,
        }
    ]


def _unready_jdt_receipt() -> dict:
    return {
        "status": "PASS",
        "diagnostics": [
            {
                "severity": 1,
                "source": "jdtls",
                "message": (
                    "Implicit super constructor Object() is undefined for default "
                    "constructor. Must define an explicit constructor"
                ),
            },
            {
                "severity": 1,
                "source": "jdtls",
                "message": "Some ordinary source diagnostic",
            },
        ],
    }


def test_recoverable_creation_conflict_is_not_promoted_to_final_guard_failure() -> None:
    assert is_recoverable_mutation_failure("MUTATION_TARGET_CREATION_CONFLICT")
    assert not _is_post_argument_semantic_failure_code(
        "MUTATION_TARGET_CREATION_CONFLICT"
    )
    assert _is_post_argument_semantic_failure_code("MUTATION_TARGET_DRIFT")


def test_existing_target_rejects_create_and_conflict_remains_recoverable() -> None:
    path = "src/main/java/example/DebugToken.java"
    context = progress_aware_tool_loop.TargetMutationContext(
        target_path=path,
        target_symbol="DebugToken",
        source_body="package example; public class DebugToken {}",
        is_new_file=False,
        evidence_source="search_code_rag",
    )
    error = progress_aware_tool_loop._mutation_target_error(
        "apply_source_edit",
        {
            "operation": "create_file",
            "path": path,
            "content": "package example; public class DebugToken {}",
        },
        context,
    )
    assert error is not None
    code = error.partition(":")[0]
    assert code == "MUTATION_TARGET_CREATION_CONFLICT"
    assert is_recoverable_mutation_failure(code)


def test_existing_verifier_repair_schema_is_host_bound_to_new_source_only() -> None:
    path = "src/main/java/example/DebugToken.java"
    state = progress_aware_tool_loop.HostRunState(
        validation_status="FAIL",
        mutation_context=progress_aware_tool_loop.TargetMutationContext(
            target_path=path,
            target_symbol="DebugToken",
            source_body="package example; public class DebugToken {}",
            is_new_file=False,
            evidence_source="verifier_workspace_source",
            writable_paths=(path,),
            target_pinned=True,
        ),
    )
    tools = progress_aware_tool_loop._constrain_verifier_repair_tools(
        (_apply_source_edit_schema(),),
        state,
    )
    parameters = tools[0]["function"]["parameters"]
    assert parameters["required"] == ["new"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"new"}

    # Operation/path authority stays host-owned; the global validation schema remains broad.
    canonical = SOURCE_EDIT_SCHEMA["properties"]["operation"]["enum"]
    assert "replace_exact" in canonical
    assert "create_file" in canonical
    assert "delete_file" in canonical


def test_jdt_core_runtime_failure_is_readiness_not_source_repair_input() -> None:
    receipt = _unready_jdt_receipt()
    errors = diagnostic_errors(receipt)
    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_WORKSPACE_NOT_READY"

    # Runtime finalization installs the repair contract. The progress loop must retire
    # this verifier instead of entering a source-repair ACT turn.
    assert progress_aware_tool_loop._verification_outcome(
        "jdt_diagnostics", {"ok": True, "result": receipt}
    ) == "UNAVAILABLE"


def test_true_authority_violation_stays_fatal() -> None:
    assert _is_post_argument_semantic_failure_code("MUTATION_TARGET_DRIFT")
    assert not is_recoverable_mutation_failure("MUTATION_TARGET_DRIFT")
