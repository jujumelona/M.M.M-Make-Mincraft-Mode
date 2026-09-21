from __future__ import annotations

from minecraft_mod_ai.source_repair_semantics import atomic_repair_scope_error
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
