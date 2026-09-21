from __future__ import annotations

from dataclasses import dataclass

from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    TargetMutationContext,
    _REPAIR_GUIDANCE_VERIFIER_DIAGNOSTIC_BYTES,
    _bind_existing_verifier_repair_call,
    _bounded_verifier_recovery_observation,
    _constrain_existing_repair_schema,
    _mutation_target_error,
    _repair_source_window,
    _rollback_non_improving_verifier_repair,
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


def _failed_diagnostics(message: str, *, line: int | None = None) -> dict:
    diagnostic = {
        "severity": 1,
        "code": "UndefinedType",
        "message": message,
    }
    if line is not None:
        diagnostic["range"] = {
            "start": {"line": line, "character": 0},
            "end": {"line": line, "character": 1},
        }
    return {
        "ok": True,
        "result": {
            "status": "FAIL",
            "diagnostics": {
                f"file:///{PATH}": [diagnostic]
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
    create_error = _mutation_target_error(
        "apply_source_edit", args, state.mutation_context
    )
    assert create_error is not None
    assert create_error.startswith("MUTATION_TARGET_CREATION_CONFLICT")

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
    source = (
        "package dev.mmm.debugfixture;\n"
        "public class DebugToken {\n"
        "    RegistryWrapper value;\n"
        "}\n"
    )
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body=source,
            is_new_file=False,
            evidence_source="mutation_receipt",
        )
    )
    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics(
            "RegistryWrapper cannot be resolved to a type",
            line=2,
        ),
        "FAIL",
    )
    first = state.take_verifier_repair_guidance()
    assert first is not None
    assert PATH in first
    assert "RegistryWrapper cannot be resolved to a type" in first
    assert "replace_exact" in first
    assert "repair_window" in first
    assert "bounded repair_window" in first
    assert '"current_source":' not in first
    assert "complete corrected source body" not in first
    assert "host-owned" in first
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


@dataclass(frozen=True)
class _RepairCall:
    name: str
    arguments: dict
    raw_arguments: str = ""


def test_verifier_repair_window_is_local_and_host_binds_old_text():
    source = (
        "package dev.mmm.debugfixture;\n"
        "public class DebugToken {\n"
        "    private int stable = 7;\n"
        "    RegistryWrapper value;\n"
        "    public int stableMethod() { return stable; }\n"
        "}\n"
    )
    state = HostRunState(
        validation_status="FAIL",
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body=source,
            is_new_file=False,
            evidence_source="verifier_workspace_source",
            writable_paths=(PATH,),
            target_pinned=True,
        ),
    )
    diagnostic = {
        "path": PATH,
        "severity": 1,
        "code": "UndefinedType",
        "message": "RegistryWrapper cannot be resolved to a type",
        "range": {
            "start": {"line": 3, "character": 4},
            "end": {"line": 3, "character": 19},
        },
    }
    state.latest_verifier_errors = (diagnostic,)
    state.repair_target_diagnostics = (diagnostic,)

    window = _repair_source_window(state)
    assert window is not None
    assert "RegistryWrapper value;" in window["old"]
    assert window["old"] != source
    assert len(window["old"]) < len(source)

    bound = _bind_existing_verifier_repair_call(
        _RepairCall(
            name="apply_source_edit",
            arguments={
                "new": window["old"].replace("RegistryWrapper", "RegistryEntry")
            },
        ),
        state,
    )
    assert bound.arguments["operation"] == "replace_exact"
    assert bound.arguments["path"] == PATH
    assert bound.arguments["old"] == window["old"]
    assert bound.arguments["count"] == 1
    assert "RegistryEntry value;" in bound.arguments["new"]


def test_existing_verifier_repair_schema_forbids_whole_file_protocol():
    schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["operation", "path"],
                "additionalProperties": False,
            },
        },
    }
    projected = _constrain_existing_repair_schema(schema, target_path=PATH)
    parameters = projected["function"]["parameters"]
    assert set(parameters["properties"]) == {"new"}
    assert parameters["required"] == ["new"]
    assert parameters["properties"]["new"]["maxLength"] == 4096
    description = parameters["properties"]["new"]["description"]
    assert "bounded source window" in description
    assert "Never emit the complete source file" in description


def test_repair_window_never_degenerates_to_entire_single_line_file():
    source = "class DebugToken { RegistryWrapper value; }"
    state = HostRunState(
        validation_status="FAIL",
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body=source,
            is_new_file=False,
            evidence_source="verifier_workspace_source",
            writable_paths=(PATH,),
            target_pinned=True,
        ),
    )
    diagnostic = {
        "path": PATH,
        "severity": 1,
        "code": "UndefinedType",
        "message": "RegistryWrapper cannot be resolved to a type",
        "range": {
            "start": {"line": 0, "character": 19},
            "end": {"line": 0, "character": 34},
        },
    }
    state.latest_verifier_errors = (diagnostic,)
    state.repair_target_diagnostics = (diagnostic,)
    window = _repair_source_window(state)
    assert window is not None
    assert window["old"] == "RegistryWrapper"
    assert window["old"] != source


def test_existing_model_replace_requires_exact_old_span():
    context = TargetMutationContext(
        target_path=PATH,
        target_symbol="DebugToken",
        source_body="class DebugToken { int value = MISSING; }",
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(PATH,),
        target_pinned=True,
    )
    error = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "new": "class DebugToken { int value = 1; }",
        },
        context,
    )
    assert error is not None
    assert error.startswith("MUTATION_ATOMIC_SPAN_REQUIRED")


def test_atomic_replace_semantic_guard_checks_resulting_java_file():
    source = (
        "package dev.mmm.debugfixture;\n"
        "public class DebugToken {\n"
        "    private int computeValue() { return MISSING; }\n"
        "    public int keepMe() { return computeValue(); }\n"
        "}\n"
    )
    context = TargetMutationContext(
        target_path=PATH,
        target_symbol="DebugToken",
        source_body=source,
        is_new_file=False,
        evidence_source="verifier_workspace_source",
        writable_paths=(PATH,),
        target_pinned=True,
    )
    old = "    private int computeValue() { return MISSING; }\n"
    safe = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": old,
            "new": "    private int computeValue() { return 1; }\n",
            "count": 1,
        },
        context,
    )
    assert safe is None

    destructive = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": old,
            "new": "",
            "count": 1,
        },
        context,
    )
    assert destructive is not None
    assert destructive.startswith("REPAIR_SEMANTIC_FOOTPRINT_VIOLATION")


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


def _failed_diagnostics_many(*messages: str) -> dict:
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
                    for message in messages
                ]
            },
        },
    }


def test_repair_progress_requires_strict_verifier_improvement():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="package dev.mmm.debugfixture; public class DebugToken { A a; B b; }",
            is_new_file=False,
            evidence_source="verifier_workspace_source",
        )
    )
    assert state.record_verification(
        "java_diagnostics",
        _failed_diagnostics_many("A cannot be resolved", "B cannot be resolved"),
        "FAIL",
    )
    assert state.record_mutation(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "B b;",
            "new": "int b = 1;",
            "count": 1,
        },
        _applied_receipt(),
    )
    assert state.repair_baseline_error_count == 2
    assert state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("A cannot be resolved"),
        "FAIL",
    )
    assert state.last_verifier_quality == "IMPROVED"

    assert state.record_mutation(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "A a;",
            "new": "C c;",
            "count": 1,
        },
        _applied_receipt(),
    )
    assert state.repair_baseline_error_count == 1
    assert not state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("C cannot be resolved"),
        "FAIL",
    )
    assert state.last_verifier_quality == "NON_IMPROVING"


def test_non_improving_repair_rolls_back_to_verifier_proven_source():
    original = (
        "package dev.mmm.debugfixture; "
        "public class DebugToken { Missing value; }"
    )
    candidate = (
        "package dev.mmm.debugfixture; "
        "public class DebugToken { OtherMissing value; }"
    )
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body=original,
            is_new_file=False,
            evidence_source="verifier_workspace_source",
        )
    )
    state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("Missing cannot be resolved"),
        "FAIL",
    )
    assert state.record_mutation(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "Missing",
            "new": "OtherMissing",
            "count": 1,
        },
        _applied_receipt(),
    )
    assert state.mutation_context is not None
    assert state.mutation_context.source_body == candidate
    assert not state.record_verification(
        "java_diagnostics",
        _failed_diagnostics("OtherMissing cannot be resolved"),
        "FAIL",
    )

    calls = []

    class Runtime:
        def call(self, stage, name, arguments):
            calls.append((stage, name, dict(arguments)))
            return {
                "schema_version": "mmm/source-patch-receipt-v1",
                "status": "APPLIED",
                "operations": [
                    {
                        "path": PATH,
                        "before_sha256": "sha256:candidate",
                        "after_sha256": "sha256:original",
                    }
                ],
            }

    assert _rollback_non_improving_verifier_repair(
        state,
        Runtime(),
        stage="generation",
    )
    assert calls == [
        (
            "generation",
            "apply_source_edit",
            {
                "operation": "replace_exact",
                "path": PATH,
                "old": candidate,
                "new": original,
                "count": 1,
            },
        )
    ]
    assert state.mutation_context is not None
    assert state.mutation_context.source_body == original
    assert state.validation_status == "FAIL"
    assert state.last_verifier_quality == "NON_IMPROVING"
    assert "Missing cannot be resolved" in str(state.latest_verifier_errors)


def test_atomic_java_repair_preserves_package_and_public_type_identity():
    source = (
        "package dev.mmm.debugfixture; "
        "public class DebugToken { int value = MISSING; "
        "public static class Builder {} }"
    )
    context = TargetMutationContext(
        target_path=PATH,
        target_symbol="DebugToken",
        source_body=source,
        is_new_file=False,
        evidence_source="verifier_workspace_source",
    )

    wrong_type = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "public class DebugToken",
            "new": "public class StarLinkMod",
            "count": 1,
        },
        context,
    )
    assert wrong_type is not None
    assert wrong_type.startswith("REPAIR_SEMANTIC_IDENTITY_VIOLATION")
    assert "DebugToken" in wrong_type

    wrong_package = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "package dev.mmm.debugfixture;",
            "new": "package other.pkg;",
            "count": 1,
        },
        context,
    )
    assert wrong_package is not None
    assert wrong_package.startswith("REPAIR_SEMANTIC_IDENTITY_VIOLATION")

    valid = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "int value = MISSING;",
            "new": "int value = 1;",
            "count": 1,
        },
        context,
    )
    assert valid is None

    default_path = "src/main/java/DebugToken.java"
    default_context = TargetMutationContext(
        target_path=default_path,
        target_symbol="DebugToken",
        source_body="class DebugToken {}",
        is_new_file=False,
        evidence_source="verifier_workspace_source",
    )
    added_package = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": default_path,
            "old": "class DebugToken {}",
            "new": "package invented.pkg; class DebugToken {}",
            "count": 1,
        },
        default_context,
    )
    assert added_package is not None
    assert added_package.startswith("REPAIR_SEMANTIC_IDENTITY_VIOLATION")


def test_atomic_java_repair_preserves_behavioral_footprint():
    current = (
        "package dev.mmm.debugfixture; "
        "public class DebugToken { "
        "public static final String ID = \"debug\"; "
        "private int computeValue(int base) { return base + MISSING; } "
        "public int upgrade(int level) { return computeValue(level); } "
        "}"
    )
    context = TargetMutationContext(
        target_path=PATH,
        target_symbol="DebugToken",
        source_body=current,
        is_new_file=False,
        evidence_source="verifier_workspace_source",
    )

    removed_method = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "private int computeValue(int base) { return base + MISSING; } ",
            "new": "",
            "count": 1,
        },
        context,
    )
    assert removed_method is not None
    assert removed_method.startswith("REPAIR_SEMANTIC_FOOTPRINT_VIOLATION")
    assert "method:computeValue" in removed_method

    repaired = _mutation_target_error(
        "apply_source_edit",
        {
            "operation": "replace_exact",
            "path": PATH,
            "old": "MISSING",
            "new": "1",
            "count": 1,
        },
        context,
    )
    assert repaired is None


def test_repair_guidance_densifies_target_diagnostics_without_uri_repetition():
    state = HostRunState(
        mutation_context=TargetMutationContext(
            target_path=PATH,
            target_symbol="DebugToken",
            source_body="package dev.mmm.debugfixture; public class DebugToken {}",
            is_new_file=False,
            evidence_source="verifier_workspace_source",
        )
    )
    diagnostics = tuple(
        {
            "path": PATH,
            "uri": "file:///very/long/workspace/root/" + PATH,
            "severity": 1,
            "line": index + 1,
            "code": "UndefinedType",
            "message": f"MissingType{index} " + ("x" * 260),
        }
        for index in range(20)
    )
    state.latest_verifier_tool = "java_diagnostics"
    state.latest_verifier_errors = diagnostics

    import json

    phase_payload = json.loads(
        _bounded_verifier_recovery_observation(
            state,
            errors=diagnostics,
        )
    )
    repair_payload = json.loads(
        _bounded_verifier_recovery_observation(
            state,
            errors=diagnostics,
            budget_bytes=_REPAIR_GUIDANCE_VERIFIER_DIAGNOSTIC_BYTES,
        )
    )

    assert len(phase_payload["diagnostics"]) < len(diagnostics)
    assert len(repair_payload["diagnostics"]) > len(phase_payload["diagnostics"])
    assert repair_payload["target_path"] == PATH
    assert all(item["path"] == PATH for item in repair_payload["diagnostics"])
    assert all("uri" not in item for item in repair_payload["diagnostics"])
    assert repair_payload["diagnostic_count"] == len(diagnostics)
