from __future__ import annotations

from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    TargetMutationContext,
    _mutation_target_error,
)

PATH = "src/main/java/dev/mmm/debugfixture/DebugToken.java"


def _applied_receipt() -> dict:
    return {
        "ok": True,
        "result": {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "operations": [
                {
                    "path": PATH,
                    "before_sha256": None,
                    "after_sha256": "sha256:created",
                }
            ],
        },
    }


def _failed_diagnostics(message: str) -> dict:
    return {
        "ok": True,
        "result": {
            "status": "FAIL",
            "diagnostics": {
                f"file:///{PATH}": [
                    {
                        "severity": 1,
                        "code": "UndefinedType",
                        "message": message,
                    }
                ]
            },
        },
    }


def test_applied_create_converts_target_to_atomic_rewrite_authority():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            is_new_file=True,
            evidence_source="task_capsule",
        )
    )
    args = {
        "operation": "create_file",
        "path": PATH,
        "content": "package dev.mmm.debugfixture; public class DebugToken {}",
    }
    assert state.record_mutation("apply_source_edit", args, _applied_receipt())
    assert state.mutation_context is not None
    assert state.mutation_context.is_new_file is False
    assert PATH in state.created_paths
    assert _mutation_target_error("apply_source_edit", args, state.mutation_context) is None

    structural_create = {
        "operation": "create_java_type",
        "path": PATH,
        "package_name": "dev.mmm.debugfixture",
        "declaration": "public final class DebugToken",
    }
    error = _mutation_target_error(
        "apply_source_edit", structural_create, state.mutation_context
    )
    assert error is not None
    assert error.startswith("MUTATION_TARGET_CREATION_CONFLICT")


def test_verifier_fail_and_diagnostics_are_owned_by_host_state():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="class DebugToken {}",
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    assert state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("RegistryWrapper cannot be resolved to a type"),
        "FAIL",
    )
    assert state.validation_status == "FAIL"
    assert state.latest_verifier_tool == "java_diagnostics"
    assert state.latest_verifier_fingerprint
    assert state.latest_verifier_errors
    assert "RegistryWrapper cannot be resolved to a type" in str(state.latest_verifier_errors)


def test_repair_guidance_tracks_verifier_fingerprint_not_message_history():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="class DebugToken {}",
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("RegistryWrapper cannot be resolved to a type"),
        "FAIL",
    )
    first = state.take_verifier_repair_guidance()
    assert first is not None
    assert PATH in first
    assert "RegistryWrapper cannot be resolved to a type" in first
    assert "replace_exact" in first
    assert "current_source" in first
    assert "OMIT old entirely" in first
    assert "live file" in first
    assert "create_file/create" not in first
    assert state.take_verifier_repair_guidance() is None

    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("Item.Settings cannot be resolved to a type"),
        "FAIL",
    )
    second = state.take_verifier_repair_guidance()
    assert second is not None
    assert second != first
    assert "Item.Settings cannot be resolved to a type" in second


def test_real_edit_invalidates_stale_verifier_fail_and_updates_source_body():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="RegistryWrapper value;",
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("RegistryWrapper cannot be resolved to a type"),
        "FAIL",
    )
    args = {
        "operation": "replace_exact",
        "path": PATH,
        "old": "RegistryWrapper",
        "new": "RegistryEntry",
    }
    receipt = {
        "ok": True,
        "result": {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "operations": [
                {
                    "path": PATH,
                    "before_sha256": "sha256:old",
                    "after_sha256": "sha256:new",
                }
            ],
        },
    }
    assert state.record_mutation("apply_source_edit", args, receipt)
    assert state.validation_status == "PENDING"
    assert state.latest_verifier_fingerprint is None
    assert state.mutation_context is not None
    assert "RegistryEntry" in (state.mutation_context.source_body or "")
