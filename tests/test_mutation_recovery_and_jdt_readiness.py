from __future__ import annotations

from minecraft_mod_ai.mutation_authority_final_guard import (
    _is_post_argument_semantic_failure_code,
    _latest_post_argument_semantic_failure,
)
from minecraft_mod_ai.mutation_failure_classification import (
    is_recoverable_mutation_failure,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    TargetMutationContext,
    _mutation_target_error,
)
from minecraft_mod_ai.validation_diagnostic_contract import diagnostic_errors


def _tool_failure(code: str, error: str = "rejected") -> list[dict[str, object]]:
    return [
        {
            "role": "tool",
            "content": {
                "ok": False,
                "failure_code": code,
                "error": error,
            },
        }
    ]


def test_recoverable_mutation_failures_have_one_shared_classification() -> None:
    for code in (
        "MUTATION_TARGET_CREATION_CONFLICT",
        "MUTATION_TARGET_MISSING",
        "MUTATION_TARGET_ALREADY_EXISTS",
        "MUTATION_ANCHOR_NOT_FOUND",
        "MUTATION_ANCHOR_AMBIGUOUS",
        "MUTATION_PRECONDITION_FAILED",
    ):
        assert is_recoverable_mutation_failure(code)
        assert not _is_post_argument_semantic_failure_code(code)

    assert not is_recoverable_mutation_failure("MUTATION_TARGET_DRIFT")
    assert not is_recoverable_mutation_failure("PATH_OUTSIDE_WRITABLE_SET")


def test_existing_repair_target_rejects_create_operation() -> None:
    context = TargetMutationContext(
        target_path="src/main/java/example/DebugToken.java",
        target_symbol="DebugToken",
        source_body="public final class DebugToken {}",
        is_new_file=False,
        evidence_source="workspace_source",
    )
    error = _mutation_target_error(
        "apply_source_edit",
        {
            "path": "src/main/java/example/DebugToken.java",
            "operation": "create_file",
        },
        context,
    )

    assert error is not None
    assert error.startswith("MUTATION_TARGET_CREATION_CONFLICT:")
    assert is_recoverable_mutation_failure(error.split(":", 1)[0])


def test_creation_conflict_remains_available_to_corrective_tool_loop() -> None:
    failure = _tool_failure(
        "MUTATION_TARGET_CREATION_CONFLICT",
        "target already exists; edit it instead of recreating it",
    )

    assert _latest_post_argument_semantic_failure(failure) is None


def test_true_mutation_authority_violation_remains_fatal() -> None:
    failure = _tool_failure(
        "MUTATION_TARGET_DRIFT",
        "requested target differs from the host-pinned target",
    )

    assert _is_post_argument_semantic_failure_code("MUTATION_TARGET_DRIFT")
    assert _latest_post_argument_semantic_failure(failure) == (
        "MUTATION_TARGET_DRIFT",
        "requested target differs from the host-pinned target",
    )
    assert _is_post_argument_semantic_failure_code(
        "MUTATION_AUTHORITY_SCOPE_VIOLATION"
    )
    assert _is_post_argument_semantic_failure_code("WRITE_SCOPE_VIOLATION")


def test_jdt_unresolved_object_is_verifier_readiness_failure() -> None:
    receipt = {
        "status": "PASS",
        "diagnostics": {
            "file:///workspace/DebugToken.java": [
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
                    "message": "The method token() is undefined for the type DebugToken",
                },
            ]
        },
    }

    errors = diagnostic_errors(receipt)

    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_WORKSPACE_NOT_READY"
    assert "core runtime symbols are unresolved" in errors[0]["message"]


def test_ordinary_java_error_is_still_source_repair_input() -> None:
    receipt = {
        "status": "PASS",
        "diagnostics": {
            "file:///workspace/DebugToken.java": [
                {
                    "severity": 1,
                    "source": "jdtls",
                    "code": "compiler.err.cant.resolve.location",
                    "message": "The method token() is undefined for the type DebugToken",
                }
            ]
        },
    }

    errors = diagnostic_errors(receipt)

    assert len(errors) == 1
    assert errors[0]["code"] == "compiler.err.cant.resolve.location"


def test_unavailable_jdt_receipt_remains_infrastructure_failure() -> None:
    receipt = {
        "status": "UNAVAILABLE",
        "error": "JDT language server did not initialize",
        "diagnostics": {},
    }

    errors = diagnostic_errors(receipt)

    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
