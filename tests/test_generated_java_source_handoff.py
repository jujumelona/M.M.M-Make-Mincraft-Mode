"""Real compiler regression for producers materialized during generator execution."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.construction_file_author import _verify_full_source
from minecraft_mod_ai.integrity_dispatcher import execute_generator_job
from minecraft_mod_ai.task_template_catalog import load_template


SOURCES = {
    "src/main/java/demo/first/Occurrence.java": """package demo.first;
public class Occurrence { public static final bridge.ListenerSet EVENT = new bridge.ListenerSet(); }
""",
    "src/main/java/demo/second/Occurrence.java": """package demo.second;
public class Occurrence { public static final bridge.ListenerSet EVENT = new bridge.ListenerSet(); }
""",
    "src/main/java/bridge/ListenerSet.java": """package bridge;
public class ListenerSet { public void register(Runnable effect) { Transport.start(effect); } }
""",
    "src/main/java/bridge/Transport.java": """package bridge;
public class Transport { public static void start(Runnable effect) { effect.run(); } }
""",
}
CANDIDATE = """package demo;
public class Feature {
 public void onInitialize() {
  demo.first.Occurrence.EVENT.register(this::effect);
  demo.second.Occurrence.EVENT.register(this::effect);
 }
 private void effect() { System.setProperty("handoff.ran", "true"); }
}
"""


def _dispatch(tmp_path, monkeypatch, *, change_initial=False, change_output=False, change_producer=False):
    leaf = "minecraft/event/callback"
    manifest = load_template(leaf)
    spec_port = next(p["name"] for p in manifest["inputs"] if p["type"] == "specification")
    out_port = next(p["name"] for p in manifest["outputs"] if p["type"] == "code_fragment")
    relative = "src/main/java/demo/Feature.java"
    prerequisite = "src/main/java/bridge/Transport.java"
    for name, source in SOURCES.items():
        if name == prerequisite:
            path = tmp_path / name
            path.parent.mkdir(parents=True)
            path.write_text(source)
    spec = {"leaf_id": leaf, "context_id": "fixture", "language": "java", "side": "COMMON",
            "target_path": relative, "operation": "CREATE_FILE", "bindings": {
                "content_dependency_paths_json": json.dumps([prerequisite])}}
    job = ArtifactJob(job_id="feature", template_id="", owner_module="feature", canonical_leaf=leaf,
                      implementation_id="fixture_generator",
                      deterministic_inputs={"_canonical_inputs": {spec_port: spec}})
    resolved = SimpleNamespace(context_id="fixture", to_dict=lambda: {"target": {"java_version": 17}})
    empty_cp = tmp_path / "empty-classpath"
    empty_cp.mkdir()
    context = {"java_classpath": [empty_cp]}
    monkeypatch.setattr("minecraft_mod_ai.fixed_template_generation._adapter_name", lambda *a: "llama_cpp")
    monkeypatch.setattr("minecraft_mod_ai.construction_runtime.prepare_construction_context",
                        lambda *a, **k: {"project_root": str(tmp_path), "symbols": {},
                                         "target": {"java_version": 17}, **context})

    def generate(_inputs, **kwargs):
        # The generator legitimately creates new producer namespaces after
        # dispatch captured the original prerequisite list.
        for name, source in SOURCES.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        if change_initial:
            path = tmp_path / prerequisite
            path.write_text(path.read_text() + "// changed after initial admission\n")
        _verify_full_source(CANDIDATE, spec, kwargs["construction_context"], package="demo", name="Feature",
            source_context=[*kwargs["source_context"], *({"path": name} for name in SOURCES if name != prerequisite)])
        if change_producer:
            path = tmp_path / "src/main/java/demo/first/Occurrence.java"
            path.write_text(path.read_text() + "// changed after compilation\n")
        return {out_port: CANDIDATE.replace('"true"', '"false"') if change_output else CANDIDATE}

    authority = SimpleNamespace(executors={"fixture_generator": generate})
    monkeypatch.setattr("minecraft_mod_ai.resolved_version_context.execution_context", lambda *a: resolved)
    monkeypatch.setattr("minecraft_mod_ai.integrity_dispatcher.verify_job_binding", lambda *a: authority)
    monkeypatch.setattr("minecraft_mod_ai.integrity_dispatcher.verify_candidate_content", lambda *a: None)
    monkeypatch.setattr("minecraft_mod_ai.integrity_dispatcher.require_registered_leaf_binding",
                        lambda *a: {"state": "not_reviewed"})
    return execute_generator_job(job, context=context, router=SimpleNamespace(), base_dir=tmp_path)


def test_post_generation_producers_and_transitive_sources_reach_final_attribution(tmp_path, monkeypatch):
    # An unrelated invalid file must never become a prerequisite merely because
    # it shares the generated listener's package.
    unrelated = tmp_path / "src/main/java/demo/Unrelated.java"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("package demo; public class Unrelated { BROKEN }")
    result = _dispatch(tmp_path, monkeypatch)
    assert result["status"] == "PASS"
    assert (tmp_path / "src/main/java/demo/Feature.java").read_text() == CANDIDATE
    receipt = next(r for r in result["validations"] if r["validator"] == "materialized_prerequisite_source")
    assert {r["path"] for r in receipt["sources"]} == set(SOURCES)
    for row in receipt["sources"]:
        assert row["sha256"] == hashlib.sha256(SOURCES[row["path"]].encode()).hexdigest()
    assert next(r for r in result["validations"] if r["validator"] == "client_side_only")["status"] == "PASS"


def test_post_generation_snapshot_does_not_refresh_away_changed_initial_receipt(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="JAVA_COMPANION_SOURCE_CHANGED"):
        _dispatch(tmp_path, monkeypatch, change_initial=True)
    assert not (tmp_path / "src/main/java/demo/Feature.java").exists()


@pytest.mark.parametrize("changes,error", [({"change_output": True}, "JAVA_COMPILED_SOURCE_CONTEXT_MISMATCH"),
                                           ({"change_producer": True}, "JAVA_COMPANION_SOURCE_CHANGED")])
def test_compilation_handoff_is_bound_to_candidate_and_producer_bytes(tmp_path, monkeypatch, changes, error):
    with pytest.raises(ValueError, match=error):
        _dispatch(tmp_path, monkeypatch, **changes)
    assert not (tmp_path / "src/main/java/demo/Feature.java").exists()


def test_companion_source_set_is_checked_for_its_own_caller_side(tmp_path):
    from minecraft_mod_ai.integrity_validators import validate_side

    name = "src/client/java/client/Screen.java"
    source = "package client; public class Screen { public static void render(){ new Screen(); } }"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text(source)
    records = [{"path": name, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "authority": "materialized_prerequisite_source"}]
    options = {"leaf_id": "minecraft/event/callback", "side": "COMMON", "classpath": [],
               "java_version": "17", "project_root": tmp_path, "source_context": records}
    assert validate_side("public class Entry {}", **options)["status"] == "PASS"
    with pytest.raises(ValueError, match="SIDE_VIOLATION"):
        validate_side("public class Entry { void run(){client.Screen.render();} }", **options)


def test_qualified_names_in_escaped_java_text_blocks_are_not_dependencies(tmp_path):
    from minecraft_mod_ai.integrity_validators import validate_side
    from minecraft_mod_ai.java_source_snapshot import materialized_java_source_context

    broken = tmp_path / "src/main/java/danger/Broken.java"
    broken.parent.mkdir(parents=True)
    broken.write_text("package danger; public class Broken { INVALID }")
    source = r'''package demo;
public class Feature {
 String description = """
   \"""
   danger.Broken.call();
   """;
}
'''
    snapshot = materialized_java_source_context(source, project_root=tmp_path,
        target_path="src/main/java/demo/Feature.java")
    assert snapshot == ()
    assert validate_side(source, leaf_id="minecraft/event/callback", side="COMMON",
                         classpath=[], java_version="17", project_root=tmp_path,
                         source_context=snapshot)["status"] == "PASS"


def test_dotted_field_access_does_not_require_an_unrelated_package_source(tmp_path):
    from minecraft_mod_ai.java_source_snapshot import materialized_java_source_context
    from minecraft_mod_ai.integrity_validators import validate_side

    broken = tmp_path / "src/main/java/demo/Helper.java"
    broken.parent.mkdir(parents=True)
    broken.write_text("package demo; public class Helper { INVALID }")
    source = """public class Entry {
 static class Box { int Helper; }
 int run() { Box demo = new Box(); return demo.Helper; }
}
"""
    snapshot = materialized_java_source_context(source, project_root=tmp_path,
        target_path="src/main/java/Entry.java")
    assert snapshot == ()
    assert validate_side(source, leaf_id="minecraft/event/callback", side="COMMON",
                         classpath=[], java_version="17", project_root=tmp_path,
                         source_context=snapshot)["status"] == "PASS"


def test_common_companion_cannot_hide_a_client_only_call(tmp_path):
    from minecraft_mod_ai.integrity_validators import validate_side

    sources = {
        "src/main/java/net/fabricmc/api/EnvType.java":
            "package net.fabricmc.api; public enum EnvType {CLIENT, SERVER}",
        "src/main/java/net/fabricmc/api/Environment.java": """package net.fabricmc.api;
@java.lang.annotation.Retention(java.lang.annotation.RetentionPolicy.CLASS)
public @interface Environment { EnvType value(); }
""",
        "src/main/java/api/Client.java": """package api;
@net.fabricmc.api.Environment(net.fabricmc.api.EnvType.CLIENT) public class Client {}
""",
        "src/main/java/demo/Helper.java": """package demo;
public class Helper { public static void run(){ new api.Client(); } }
""",
    }
    records = []
    for name, code in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        records.append({"path": name, "sha256": hashlib.sha256(code.encode()).hexdigest(),
                        "authority": "materialized_prerequisite_source"})
    with pytest.raises(ValueError, match="SIDE_VIOLATION"):
        validate_side("public class Entry { void run(){demo.Helper.run();} }",
                      leaf_id="minecraft/event/callback", side="COMMON", classpath=[], java_version="17",
                      project_root=tmp_path, source_context=records)
