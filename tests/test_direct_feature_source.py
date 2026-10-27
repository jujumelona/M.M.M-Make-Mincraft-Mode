"""Requested gameplay must be written even when its implementation is absent."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess

import pytest

from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


def _fixture():
    behavior = {
        "behavior_id": "new_gameplay", "intent": "Claim an eligible location",
        "trigger": "Player reaches a location and initiates a claim",
        "effects": "Remember the location as claimed",
        "source_quote": "Allow claiming locations",
        "runtime_state_reads": ["eligible locations"],
        "runtime_state_writes": ["claimed locations"],
    }
    spec = {
        "language": "java", "target_path": "src/main/java/example/Feature.java",
        "requirement": behavior["source_quote"], "side": "COMMON",
        "bindings": {"package_name": "example", "class_name": "Feature",
                     "request_feature_entrypoint": True,
                     "construction_behaviors_json": json.dumps([behavior])},
    }
    context = {
        "host_direct_feature_source": True,
        "authoring_contract": {},
        "symbols": {name: {"kind": "CLASS", "owner": name, "access": 1}
                    for name in ["example.GameEvents", "net.fabricmc.api.ModInitializer"]},
    }
    return behavior, spec, context


class SourceCoder:
    """Deterministic model transport; Java planning/rendering remains production code."""

    units = [
        {"kind": "field", "name": "claimed", "purpose": "Store claimed location ids"},
        {"kind": "method", "name": "claim", "purpose": "Record a new eligible location"},
        {"kind": "method", "name": "claimCount", "purpose": "Expose the number of claimed locations"},
    ]

    def __init__(self):
        self.body_tasks = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        if data_name == "submit_java_construction_plan_count":
            return {"count": len(self.units)}
        match = re.fullmatch(r"submit_java_construction_plan_(\d+)_value_(kind|name|purpose)", data_name)
        if match:
            return {match[2]: self.units[int(match[1]) - 1][match[2]]}
        if data_name == "submit_java_declaration_1":
            return {"visibility": "private", "static": False, "final": True,
                    "type": "java.util.Set<String>"}
        if data_name.startswith("submit_java_declaration_"):
            index = int(data_name.rsplit("_", 1)[1])
            return {"visibility": "public", "static": False,
                    "return_type": "void" if index == 2 else "int",
                    "type_parameters": "", "parameters": (
                        [{"type": "String", "name": "location"}] if index == 2 else []),
                    "throws": [], "annotations": []}
        raise AssertionError("Direct coding entered an existence/proof decision: " + data_name)

    def generate_java_source(self, role, messages, *, data_name, source_frame):
        task = json.loads(messages[-1]["content"])
        self.body_tasks.append(task)
        signature = task["host_signature"]
        if " claimed" in signature:
            body = "new java.util.HashSet<>()"
        elif " claim(" in signature:
            body = "if (location != null && !location.isBlank()) this.claimed.add(location);"
        elif " claimCount(" in signature:
            body = "return this.claimed.size();"
        elif "registerBehavior" in signature:
            body = "example.GameEvents.register(location -> this.claim(location));"
        else:
            raise AssertionError(signature)
        return source_frame["prefix"] + body + source_frame["suffix"]


def _run_java(tmp_path, source):
    java = shutil.which("java")
    if java is None:
        pytest.skip("Java runtime is required for the generated-code harness")
    sources = {
        "example/Feature.java": source,
        "net/fabricmc/api/ModInitializer.java": (
            "package net.fabricmc.api; public interface ModInitializer {void onInitialize();}"),
        "example/GameEvents.java": (
            "package example; public class GameEvents {"
            "private static java.util.function.Consumer<String> listener;"
            "public static void register(java.util.function.Consumer<String> value) {listener=value;}"
            "public static void fire(String value) {listener.accept(value);}}"),
        "example/Harness.java": (
            "package example; public class Harness {public static void main(String[] args) {"
            "Feature feature=new Feature(); feature.onInitialize();"
            "GameEvents.fire(\"alpha\"); GameEvents.fire(\"alpha\"); GameEvents.fire(\"\");"
            "if(feature.claimCount()!=1) throw new AssertionError(\"first claim\");"
            "GameEvents.fire(\"beta\");"
            "if(feature.claimCount()!=2) throw new AssertionError(\"second claim\");"
            "System.out.print(\"claims=2\");}}"),
    }
    files = []
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        files.append(str(path))
    classes = tmp_path / "classes"
    compiled = subprocess.run(
        [java, "-m", "jdk.compiler/com.sun.tools.javac.Main", "-d", str(classes), *files],
        capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([java, "-cp", str(classes), "example.Harness"],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == "claims=2"


@pytest.mark.parametrize("trigger", [
    "Player reaches a location and initiates a claim",
    "A location becomes eligible for automatic claiming",
])
def test_direct_source_authors_new_state_and_event_code_before_any_existing_site(trigger, tmp_path):
    behavior, spec, context = _fixture()
    behavior["trigger"] = trigger
    coder = SourceCoder()
    source = compose_first_pass_java(
        coder, spec, [], construction_context=context, behaviors=[behavior])
    _run_java(tmp_path, source)
    assert len(coder.body_tasks) == 4
    registrar_task = coder.body_tasks[-1]
    assert registrar_task["assigned_behavior"]["trigger"] == trigger
    assert any("claim(String location)" in row["signature"]
               for row in registrar_task["declared_member_api"])


def test_direct_registrar_name_uses_the_canonical_behavior_identifier():
    behavior, spec, context = _fixture()
    behavior["behavior_id"] = " new_gameplay "
    expected = "registerBehavior" + hashlib.sha256(b"new_gameplay").hexdigest()[:20]
    source = compose_first_pass_java(SourceCoder(), spec, [], construction_context=context, behaviors=[behavior])
    assert "private void " + expected + "()" in source
    assert "this." + expected + "();" in source


def test_production_entrypoint_does_not_run_capability_or_proof_planning(tmp_path, monkeypatch):
    from test_incremental_java_pipeline import production_fixture
    from test_incremental_java_author import OperationCoder
    build = production_fixture(tmp_path, monkeypatch)
    coder = OperationCoder()
    source = build(coder)
    count = len(coder.requests) + len(coder.expression_requests)
    assert build(coder) == source
    assert len(coder.requests) + len(coder.expression_requests) == count


def test_direct_registration_compiler_error_is_sent_back_to_the_source_writer():
    from minecraft_mod_ai.direct_feature_repair import repair_direct_feature_member

    behavior, _, _ = _fixture()
    name = "registerBehavior" + hashlib.sha256(behavior["behavior_id"].encode()).hexdigest()[:20]
    source = "package example;\npublic class Feature {\nprivate void " + name + "() {\nmissing();\n}\n}\n"

    class RepairCoder:
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            assert "cannot find symbol" in task["javac_diagnostic"]
            assert "missing();" in task["current_member"]
            return source_frame["prefix"] + "example.GameEvents.register(value -> {});" + source_frame["suffix"]

    repaired = repair_direct_feature_member(
        RepairCoder(), source, "Feature.java:4: error: cannot find symbol\nmissing();",
        context={"filename": "Feature.java", "target_path": "src/main/java/example/Feature.java",
                 "symbols": {}, "owner": "example.Feature", "api_evidence": {},
                 "attempt": 1, "behaviors": [behavior]})
    assert "missing();" not in repaired
    assert "example.GameEvents.register" in repaired


def test_direct_field_initializer_can_be_repaired_without_rewriting_the_class():
    from minecraft_mod_ai.direct_feature_repair import repair_direct_feature_member

    source = ("package example;\npublic class Feature {\n"
              "private final java.util.Set<String> claimed = missing();\n"
              "public int claimCount() {return claimed.size();}\n}\n")

    class RepairCoder:
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            assert "missing();" in task["current_member"]
            assert source_frame["kind"] == "field"
            return source_frame["prefix"] + "new java.util.HashSet<>()" + source_frame["suffix"]

    repaired = repair_direct_feature_member(
        RepairCoder(), source, "Feature.java:3: error: cannot find symbol\nmissing();",
        context={"filename": "Feature.java", "target_path": "src/main/java/example/Feature.java",
                 "symbols": {}, "owner": "example.Feature", "api_evidence": {}, "attempt": 1})
    assert "private final java.util.Set<String> claimed = new java.util.HashSet<>();" in repaired
    assert "public int claimCount() {return claimed.size();}" in repaired


def test_direct_repair_and_no_progress_retry_keep_behavior_and_existing_ui_contract():
    from minecraft_mod_ai.direct_feature_repair import repair_direct_feature_member

    behavior, _, _ = _fixture()
    name = "registerBehavior" + hashlib.sha256(behavior["behavior_id"].encode()).hexdigest()[:20]
    source = "package example;\npublic class Feature {\nprivate void " + name + "() {\nmissing();\n}\n}\n"
    requirements = {"original_feature_goal": "Claim eligible locations",
                    "implementation_requirements": "Keep the existing selected-location action",
                    "retained_typed_behavior": {"ui_owned_records": [{"action": "select_location"}]},
                    "owned_runtime_records": [{"name": "selected_location"}]}
    binding = {"selected_location": {"binding": "typed_input"}}

    class RepairCoder:
        calls = 0

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            self.calls += 1
            task = json.loads(messages[-1]["content"])
            assert task["assigned_behaviors"] == [behavior]
            assert task["runtime_state_bindings"] == binding
            for key, value in requirements.items():
                assert task[key] == value
            if self.calls == 1:
                return source_frame["prefix"] + "missing();" + source_frame["suffix"]
            return source_frame["prefix"] + "example.GameEvents.register(value -> {});" + source_frame["suffix"]

    coder = RepairCoder()
    repaired = repair_direct_feature_member(
        coder, source, "Feature.java:4: error: cannot find symbol\nmissing();",
        context={"filename": "Feature.java", "target_path": "src/main/java/example/Feature.java",
                 "symbols": {}, "owner": "example.Feature", "api_evidence": {},
                 "attempt": 1, "behaviors": [behavior], "runtime_state_bindings": binding,
                 "implementation_context": requirements})
    assert "missing();" not in repaired
    assert coder.calls == 2


def test_direct_compile_includes_existing_qualified_ui_source_without_proof_receipts(tmp_path):
    from minecraft_mod_ai.java_source_snapshot import materialized_java_sources

    files = {
        "src/main/java/example/actions/Occurrence.java": (
            "package example.actions; public class Occurrence {"
            "public static void claim() {example.inputs.Selection.read();}}"),
        "src/main/java/example/inputs/Selection.java": (
            "package example.inputs; public class Selection {public static void read() {}}"),
        "src/main/java/example/unused/Decoy.java": (
            "package example.unused; public class Decoy {}"),
        "src/client/java/example/client/ClientOnly.java": (
            "package example.client; public class ClientOnly {}"),
    }
    for name, code in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
    source = ("package example; public class Feature {"
              "void start() {example.actions.Occurrence.claim();}"
              "/* example.unused.Decoy.fake(); */"
              "String note=\"example.unused.Decoy.fake();\";}")
    companions = materialized_java_sources(
        source, project_root=tmp_path, target_path="src/main/java/example/Feature.java",
        include_qualified_sources=True)
    assert set(companions) == {
        "src/main/java/example/actions/Occurrence.java",
        "src/main/java/example/inputs/Selection.java",
    }


def test_production_direct_compiler_stages_qualified_dependency_without_receipts(tmp_path, monkeypatch):
    from minecraft_mod_ai.construction_project_compile import verify_java_source_with_dependencies

    java = shutil.which("java")
    if java is None:
        pytest.skip("Java runtime is required for the compiler harness")
    jdk = tmp_path / "jdk"
    compiler = jdk / "bin/javac"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\nexec " + shlex.quote(java) + " -m jdk.compiler/com.sun.tools.javac.Main \"$@\"\n")
    compiler.chmod(0o755)
    monkeypatch.setattr("minecraft_mod_ai.java_lsp._resolve_project_java_home", lambda *a, **kw: str(jdk))
    dependency = tmp_path / "src/main/java/example/actions/Occurrence.java"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("package example.actions; public class Occurrence {public static void claim() {}}")
    source = "package example; public class Feature {void start() {example.actions.Occurrence.claim();}}"
    context = {"project_root": str(tmp_path), "target": {"java_version": 17},
               "java_classpath": [str(tmp_path / "empty-classpath")]}
    kwargs = {"target_path": "src/main/java/example/Feature.java", "filename": "Feature.java"}
    with pytest.raises(ValueError, match="does not exist"):
        verify_java_source_with_dependencies(source, context=context, **kwargs)
    context["host_direct_feature_source"] = True
    companions = verify_java_source_with_dependencies(source, context=context, **kwargs)
    assert set(companions) == {"src/main/java/example/actions/Occurrence.java"}
