from __future__ import annotations

import json

from types import SimpleNamespace

from minecraft_mod_ai.model_adapters import ToolCall
from minecraft_mod_ai.mutation_authority import (
    CURRENT_MUTATION_AUTHORITY,
    MutationAuthority,
)
from minecraft_mod_ai.mutation_failure_classification import (
    is_post_argument_semantic_failure_code as _is_post_argument_semantic_failure_code,
    is_recoverable_mutation_failure,
    latest_post_argument_semantic_failure as _latest_post_argument_semantic_failure,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    TargetMutationContext,
    _bind_host_owned_existing_source_call,
    _implementation_obligation_has_progress,
    _mutation_target_error,
    _recover_creation_conflict_target,
    _source_edit_schema_for_context,
)
from minecraft_mod_ai.source_repair_semantics import (
    existing_java_structurally_subsumes_candidate,
)
from minecraft_mod_ai.verifier_repair_admission_recovery import (
    recover_schema_rejected_host_bound_existing_calls,
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


def test_creation_conflict_rebinds_live_file_for_small_coder(tmp_path) -> None:
    target_path = "src/main/java/com/example/starforge/StarForgeMod.java"
    target = tmp_path / target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    source = (
        "package com.example.starforge;\n"
        "public final class StarForgeMod {\n"
        "    public static int credits() { return 1; }\n"
        "}\n"
    )
    target.write_text(source, encoding="utf-8")

    state = HostRunState()
    state.unchanged_mutation_fingerprints.add("failed-create")
    state.unapplied_mutation_fixed_point = True
    state.semantic_fixed_point = True

    recovered = _recover_creation_conflict_target(
        state,
        SimpleNamespace(workspace_root=str(tmp_path)),
        {"operation": "create_file", "path": target_path},
    )

    assert recovered is not None
    assert recovered.target_path == target_path
    assert recovered.is_new_file is False
    assert recovered.target_pinned is True
    assert recovered.is_mutation_ready is True
    assert recovered.source_body == source
    assert state.mutation_context == recovered
    assert state.unapplied_mutation_fixed_point is False
    assert state.semantic_fixed_point is False
    assert state.unchanged_mutation_fingerprints == set()


def test_existing_rebound_target_schema_removes_create_and_pins_path() -> None:
    target_path = "src/main/java/com/example/starforge/StarForgeMod.java"
    schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "edit source",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create_file", "replace_exact", "insert_after"],
                    },
                    "path": {"type": "string"},
                    "target_path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["operation", "path"],
            },
        },
    }
    context = TargetMutationContext(
        target_path=target_path,
        source_body="public final class StarForgeMod {}",
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(target_path,),
        target_pinned=True,
    )

    projected = _source_edit_schema_for_context(schema, context)
    properties = projected["function"]["parameters"]["properties"]

    assert properties["operation"]["enum"] == ["replace_exact", "insert_after"]
    assert properties["path"]["enum"] == [target_path]
    assert "target_path" not in properties


def test_unpinned_authored_schema_constrains_destination_to_host_roots() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "edit source",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                    "target_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["operation", "path"],
            },
        },
    }
    root = "src/main/java/ai/minecraft/generated/authored_demo/"
    authority = MutationAuthority.bounded_roots((root, "src/main/resources/"))
    token = CURRENT_MUTATION_AUTHORITY.set(authority)
    try:
        projected = _source_edit_schema_for_context(schema, None)
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(token)

    parameters = projected["function"]["parameters"]
    properties = parameters["properties"]
    assert "target_path" not in properties
    assert properties["path"]["pattern"].startswith("^(?:")
    assert root in properties["path"]["description"]
    assert parameters["additionalProperties"] is False


def test_recovered_authored_existing_target_exposes_only_updated_source() -> None:
    target_path = (
        "src/main/java/ai/minecraft/generated/authored_demo/StarForgeMod.java"
    )
    source = (
        "package ai.minecraft.generated.authored_demo;\n"
        "public final class StarForgeMod {\n"
        "    public static int credits() { return 1; }\n"
        "}\n"
    )
    schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "edit source",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create_file", "replace_exact", "insert_after"],
                    },
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["operation", "path"],
            },
        },
    }
    context = TargetMutationContext(
        target_path=target_path,
        source_body=source,
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(target_path,),
        target_pinned=True,
    )
    authority = MutationAuthority.bounded_roots(
        ("src/main/java/ai/minecraft/generated/authored_demo/",)
    )
    token = CURRENT_MUTATION_AUTHORITY.set(authority)
    try:
        projected = _source_edit_schema_for_context(schema, context)
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(token)

    parameters = projected["function"]["parameters"]
    assert set(parameters["properties"]) == {"new"}
    assert parameters["required"] == ["new"]
    assert parameters["additionalProperties"] is False


def test_exact_task_existing_target_exposes_only_model_owned_new_source() -> None:
    target_path = (
        "src/main/java/ai/minecraft/generated/authored_demo/AuthoredFeature001.java"
    )
    source = (
        "package ai.minecraft.generated.authored_demo;\n"
        "public final class AuthoredFeature001 {\n"
        "    public static void initialize() {}\n"
        "}\n"
    )
    schema = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "edit source",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create_file", "replace_exact", "insert_after"],
                    },
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "count": {"type": "integer"},
                    "content": {"type": "string"},
                },
                "required": ["operation", "path"],
            },
        },
    }
    context = TargetMutationContext(
        target_path=target_path,
        source_body=source,
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(target_path,),
        creatable_paths=(),
        target_pinned=True,
    )
    authority = MutationAuthority.exact((target_path,), task_id="authored_feature_001")
    token = CURRENT_MUTATION_AUTHORITY.set(authority)
    try:
        projected = _source_edit_schema_for_context(schema, context)
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(token)

    parameters = projected["function"]["parameters"]
    assert set(parameters["properties"]) == {"new"}
    assert parameters["required"] == ["new"]
    assert parameters["additionalProperties"] is False
    assert "create_file" not in json.dumps(projected, ensure_ascii=False)


def test_recovered_authored_existing_target_binds_exact_live_old_source() -> None:
    target_path = (
        "src/main/java/ai/minecraft/generated/authored_demo/StarForgeMod.java"
    )
    source = (
        "package ai.minecraft.generated.authored_demo;\n"
        "public final class StarForgeMod {\n"
        "    public static int credits() { return 1; }\n"
        "}\n"
    )
    updated = source.replace("return 1", "return 2")
    state = HostRunState()
    state.mutation_context = TargetMutationContext(
        target_path=target_path,
        source_body=source,
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(target_path,),
        target_pinned=True,
    )
    call = ToolCall(
        id="edit-1",
        name="apply_source_edit",
        arguments={"new": updated},
        raw_arguments='{"new":"model-only"}',
    )
    authority = MutationAuthority.bounded_roots(
        ("src/main/java/ai/minecraft/generated/authored_demo/",)
    )
    token = CURRENT_MUTATION_AUTHORITY.set(authority)
    try:
        bound = _bind_host_owned_existing_source_call(call, state)
        assert bound.arguments == {
            "operation": "replace_exact",
            "path": target_path,
            "old": source,
            "new": updated,
            "count": 1,
        }
        assert (
            _mutation_target_error(
                bound.name,
                bound.arguments,
                state.mutation_context,
            )
            is None
        )
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(token)


def test_rejected_existing_source_call_drops_redundant_host_fields() -> None:
    target_path = (
        "src/main/java/ai/minecraft/generated/authored_demo/StarForgeMod.java"
    )
    source = (
        "package ai.minecraft.generated.authored_demo;\n"
        "public final class StarForgeMod {\n"
        "    public static int credits() { return 1; }\n"
        "}\n"
    )
    context = TargetMutationContext(
        target_path=target_path,
        source_body=source,
        is_new_file=False,
        evidence_source="workspace_existing_target",
        writable_paths=(target_path,),
        target_pinned=True,
    )
    rejected = ToolCall(
        id="rejected-1",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_SCHEMA_INVALID",
            "original_tool": "apply_source_edit",
            "raw_arguments": (
                '{"new":'
                + __import__("json").dumps(source.replace("return 1", "return 2"))
                + ',"path":'
                + __import__("json").dumps(target_path)
                + '}'
            ),
        },
    )

    recovered = recover_schema_rejected_host_bound_existing_calls(
        (rejected,),
        phase="ACT",
        validation_status="PENDING",
        context=context,
    )

    assert recovered is not None
    assert recovered[0].name == "apply_source_edit"
    assert recovered[0].arguments == {
        "new": source.replace("return 1", "return 2")
    }


def test_existing_java_structural_superset_preserves_richer_source() -> None:
    path = "src/main/java/ai/minecraft/generated/authored_demo/StellarForgeMod.java"
    current = (
        "package ai.minecraft.generated.authored_demo;\n"
        "import net.fabricmc.api.ModInitializer;\n"
        "import net.minecraft.registry.RegistryKeys;\n"
        "import org.slf4j.Logger;\n"
        "public class StellarForgeMod implements ModInitializer {\n"
        '    public static final String MOD_ID = "stellarforge";\n'
        "    public static final Logger LOGGER = null;\n"
        "    public void onInitialize() {}\n"
        "}\n"
    )
    reductive = (
        "package ai.minecraft.generated.authored_demo;\n"
        "import net.fabricmc.api.ModInitializer;\n"
        "import org.slf4j.Logger;\n"
        "public class StellarForgeMod implements ModInitializer {\n"
        "    private static final Logger LOGGER = null;\n"
        "    public void onInitialize() {}\n"
        "}\n"
    )
    additive = reductive.replace(
        "    public void onInitialize() {}\n",
        "    public void onInitialize() {}\n"
        "    public void launchShip() {}\n",
    )

    assert existing_java_structurally_subsumes_candidate(
        path,
        current,
        reductive,
    )
    assert not existing_java_structurally_subsumes_candidate(
        path,
        current,
        additive,
    )


def test_preserved_existing_source_counts_as_verified_fragment_progress() -> None:
    state = HostRunState()
    assert not _implementation_obligation_has_progress(state)
    state.preserved_existing_source = True
    assert _implementation_obligation_has_progress(state)


def test_authored_java_create_rejects_package_path_mismatch() -> None:
    authority = MutationAuthority.bounded_roots(
        ("src/main/java/ai/minecraft/generated/authored_demo/",)
    )
    token = CURRENT_MUTATION_AUTHORITY.set(authority)
    try:
        error = _mutation_target_error(
            "apply_source_edit",
            {
                "operation": "create_file",
                "path": (
                    "src/main/java/ai/minecraft/generated/authored_demo/"
                    "ship/SpaceShip.java"
                ),
                "content": (
                    "package com.example.voidmode;\n"
                    "public final class SpaceShip {}\n"
                ),
            },
            None,
        )
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(token)

    assert error is not None
    assert error.startswith("MUTATION_JAVA_PACKAGE_MISMATCH:")


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
