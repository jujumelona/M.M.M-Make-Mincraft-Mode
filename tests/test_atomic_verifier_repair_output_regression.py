from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters.base import ToolCall
from minecraft_mod_ai.source_repair_semantics import atomic_repair_scope_error
from minecraft_mod_ai.verifier_repair_admission_recovery import (
    recover_schema_rejected_verifier_repair_calls,
)
from minecraft_mod_ai.verifier_repair_window import (
    normalize_model_repair_replacement,
    repair_replacement_max_chars,
    select_verifier_repair_window,
)


def test_compiler_package_error_localizes_inline_import_and_downprojects_legacy_source():
    source = (
        "package dev.mmm.debugfixture; "
        "import net.minecraft.item.Item; "
        "public final class DebugToken {}\n"
    )
    diagnostic = {
        "severity": 1,
        "source": "javac",
        "code": "javac:error:2",
        "message": "package net.minecraft.item does not exist",
    }

    window = select_verifier_repair_window(source, (diagnostic,))

    assert window is not None
    assert window["old"].strip() == "import net.minecraft.item.Item;"
    corrected_source = (
        "package dev.mmm.debugfixture; public final class DebugToken {}\n"
    )
    assert (
        normalize_model_repair_replacement(
            source,
            window["old"],
            corrected_source,
        ).strip()
        == ""
    )


def test_production_import_span_rejects_file_level_replacement_before_runtime():
    source = (
        "package com.mineverse.shipbuilding.api;\n"
        "import net.minecraft.command.argument.ResourceLocationArgumentType;\n"
        "public class ShipBuildingConfig {}\n"
    )
    diagnostic = {
        "severity": 1,
        "message": (
            "The import net.minecraft.command.argument."
            "ResourceLocationArgumentType cannot be resolved"
        ),
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 67},
        },
    }
    window = select_verifier_repair_window(source, (diagnostic,))
    assert window is not None
    assert window["old"].startswith("import net.minecraft.command.argument.")
    max_chars = repair_replacement_max_chars(window["old"])
    assert max_chars < 4096

    file_level_chunk = (
        "package com.mineverse.shipbuilding.api;\n"
        "import net.minecraft.item.Item;\n"
        "public class ShipBuildingConfig {\n"
        + ("    int value;\n" * 30)
        + "}\n"
    )
    error = atomic_repair_scope_error(
        old_text=window["old"],
        new_text=file_level_chunk,
        max_chars=max_chars,
    )
    assert error is not None
    assert error.startswith("REPAIR_ATOMIC_REPLACEMENT_TOO_LARGE")



def test_unrelated_whole_source_rewrite_cannot_become_an_import_deletion():
    old_import = "import net.minecraft.command.argument.ResourceLocationArgumentType;\n"
    source = (
        "package com.authored.space;\n"
        + old_import
        + "public class SpaceMod {\n"
        + ("    int value;\n" * 30)
        + "}\n"
    )
    regenerated_file = (
        "package com.authored.space;\n\n"
        "import net.fabricmc.api.ModInitializer;\n"
        "public class SpaceMod implements ModInitializer {\n"
        "    @Override public void onInitialize() {}\n"
        "}\n"
    )
    rejected = ToolCall(
        id="call-rejected",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_SCHEMA_INVALID",
            "original_tool": "apply_source_edit",
            "raw_arguments": json.dumps({"new": regenerated_file}),
            "error": "tool 'apply_source_edit' emitted schema-invalid arguments at new: too long",
        },
    )
    context = SimpleNamespace(
        source_body=source,
        is_new_file=False,
        evidence_source="verifier_workspace_source",
    )

    recovered = recover_schema_rejected_verifier_repair_calls(
        (rejected,),
        phase="ACT",
        validation_status="FAIL",
        context=context,
        repair_window={"old": old_import},
    )

    assert recovered is None


def test_import_projection_preserves_coupled_type_rename():
    old = "import net.minecraft.server.network.ServerPlayerEntity;\n"
    source = (
        "package demo;\n" + old
        + "public class Demo { ServerPlayerEntity player; }\n"
    )
    candidate = source.replace(
        "net.minecraft.server.network.ServerPlayerEntity", "net.minecraft.server.level.ServerPlayer"
    ).replace("ServerPlayerEntity player", "ServerPlayer player")
    # Applying just an import deletion would lose the authored type migration.
    assert normalize_model_repair_replacement(source, old, candidate) == candidate


def test_schema_rejected_unrelated_non_import_rewrite_remains_rejected():
    old_text = "    int value;\n"
    source = "package demo;\npublic class Demo {\n" + old_text + "}\n"
    regenerated_file = "package other;\npublic class Other {}\n"
    rejected = ToolCall(
        id="call-rejected",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_SCHEMA_INVALID",
            "original_tool": "apply_source_edit",
            "raw_arguments": json.dumps({"new": regenerated_file}),
            "error": "tool 'apply_source_edit' emitted schema-invalid arguments at new: too long",
        },
    )
    context = SimpleNamespace(
        source_body=source,
        is_new_file=False,
        evidence_source="verifier_workspace_source",
    )

    assert recover_schema_rejected_verifier_repair_calls(
        (rejected,),
        phase="ACT",
        validation_status="FAIL",
        context=context,
        repair_window={"old": old_text},
    ) is None


def test_schema_rejected_repair_survives_recovery_evidence_source_rebinding():
    old = "    private static Block FEATURE_BLOCK = null;\n"
    source = (
        "package demo;\n"
        "public class Demo {\n"
        + old
        + "    public static void initialize() {}\n"
        "}\n"
    )
    corrected = source.replace(old, "")
    rejected = ToolCall(
        id="call-rejected-rag",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_SCHEMA_INVALID",
            "original_tool": "apply_source_edit",
            "raw_arguments": json.dumps({"new": corrected}),
            "error": "new is longer than the projected repair schema",
        },
    )
    context = SimpleNamespace(
        source_body=source,
        is_new_file=False,
        evidence_source="search_code_rag",
    )
    recovered = recover_schema_rejected_verifier_repair_calls(
        (rejected,),
        phase="ACT",
        validation_status="FAIL",
        context=context,
        repair_window={"old": old},
    )
    assert recovered is not None
    assert recovered[0].arguments == {"new": ""}

def test_host_authored_surface_localizes_class_modifier_not_reported_line():
    source = (
        "package dev.mmm;\n"
        "public class AuthoredFeature001 {\n"
        "    public static void initialize() {}\n"
        "}\n"
    )
    diagnostic = {
        "severity": 1,
        "source": "host-authored-contract",
        "code": "host:authored-surface",
        "line": 1,
        "message": "Host integration requires public final class AuthoredFeature001.",
    }

    window = select_verifier_repair_window(source, (diagnostic,))

    assert window is not None
    assert window["old"] == "public class AuthoredFeature001"
    assert window["start_line"] == 2


def test_schema_rejected_host_surface_repair_drops_redundant_host_fields():
    target = "src/main/java/dev/mmm/AuthoredFeature001.java"
    old_decl = "public class AuthoredFeature001"
    source = (
        "package dev.mmm;\n"
        + old_decl
        + " {\n"
        "    public static void initialize() {}\n"
        "}\n"
    )
    rejected = ToolCall(
        id="call-rejected-host-surface",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_SCHEMA_INVALID",
            "original_tool": "apply_source_edit",
            "raw_arguments": json.dumps(
                {
                    "operation": "replace_exact",
                    "path": target,
                    "old": old_decl,
                    "new": "public final class AuthoredFeature001",
                    "count": 1,
                }
            ),
            "error": "redundant host-owned fields",
        },
    )
    context = SimpleNamespace(
        source_body=source,
        target_path=target,
        is_new_file=False,
        evidence_source="mutation_receipt",
    )

    recovered = recover_schema_rejected_verifier_repair_calls(
        (rejected,),
        phase="ACT",
        validation_status="FAIL",
        context=context,
        repair_window={"old": old_decl},
    )

    assert recovered is not None
    assert recovered[0].name == "apply_source_edit"
    assert recovered[0].arguments == {"new": "public final class AuthoredFeature001"}
