"""Compile the actual source UI handoff against the pinned Minecraft/Fabric SDK."""
from copy import deepcopy
import json
import os
import shutil
import subprocess

import pytest

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.source_ui_actions import source_ui_callback
from minecraft_mod_ai.source_ui_actions_java import render_source_ui_action_files
from minecraft_mod_ai.typed_action_java import render_action_dispatch_java
from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
from minecraft_mod_ai.typed_plan_java import render_typed_plan_java
from test_incremental_java_value_integration import literal, local
from test_incremental_java_values import ValueCoder, member_named
from test_source_ui_production import source_capabilities, source_sections


def source_ui_sdk_sources():
    sections = source_sections()
    record = sections["resources_and_ui"]["specification"]["interactions"][0]
    capabilities = source_capabilities()
    plan = author_typed_plan_ir(None, "Store the submitted amount", sections, capabilities,
                               deterministic_module_kinds=())
    callback = {**source_ui_callback(record, package="demo"), "source_ui_record": deepcopy(record)}
    spec = {"target_path": "src/main/java/demo/generated/Feature.java", "bindings": {
        "package_name": "demo.generated", "class_name": "Feature", "parent_requirement_id": "selection",
        "source_ui_package_name": "demo", "source_ui_callbacks_json": json.dumps([callback])}}
    steps = [
        ("kind", {"op": "field"}), ("field", {"type": "int", "purpose": "Store the submitted amount",
                                              "static": False, "final": False}), *literal(0),
        ("kind", {"op": "assign"}), ("assign", {"operator": "="}),
        ("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "this"}),
        ("field_member", member_named("mmmField1")), *local("input_0"),
        ("kind", {"op": "return"}), ("value_kind", {"kind": "field"}),
        ("field_receiver", {"receiver_kind": "this"}), ("field_member", member_named("mmmField1")),
        ("kind", {"op": "complete"}),
    ]
    builder = IncrementalJavaBuilder(ValueCoder(steps), spec, [], {"symbols": {}}, [{
        "behavior_id": "selection.submit", "intent": "Store the original UI action's submitted amount"}])
    feature_source, _ = builder.finish()
    files = render_source_ui_action_files(plan, package="demo", minecraft_version="26.2")
    files.update({
        "src/main/java/demo/AuthoredProgram.java": render_typed_plan_java(
            plan, package="demo", minecraft_version="26.2", capabilities=capabilities),
        "src/main/java/demo/AuthoredActions.java": render_action_dispatch_java(plan, package="demo"),
        spec["target_path"]: feature_source,
    })
    return files


def test_seeded_source_ui_feature_and_original_dispatch_compile_against_actual_sdk25(tmp_path):
    # Keep command registration and both real initializers; only the listener body is model-authored.
    sources = source_ui_sdk_sources()
    classpath = os.environ.get("MMM_ACTION_COMPILE_CLASSPATH")
    if not classpath:
        pytest.skip("Exact Minecraft/Fabric SDK classpath required (MMM_ACTION_COMPILE_CLASSPATH)")
    javac = shutil.which("javac")
    assert javac, "JDK 25 compiler required for the exact Minecraft/Fabric SDK"
    paths = []
    for name, source in sources.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        paths.append(str(target))
    compiled = subprocess.run([javac, "--release", "25", "-proc:none", "-cp", classpath,
        "-d", str(tmp_path / "classes"), *paths], capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    assert (tmp_path / "classes/demo/AuthoredSourceActions.class").is_file()
    assert (tmp_path / "classes/demo/generated/Feature.class").is_file()
