"""Production host must create an original active occurrence before its listener."""
import json

import pytest

from minecraft_mod_ai.feature_registration_proof import (
    validate_registered_event_sources,
)
from minecraft_mod_ai.feature_source_work_graph import build_source_work_graph
from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


def _fixture(tmp_path, monkeypatch, *, fail_body=False):
    behavior = {
        "behavior_id": "allocate", "requirement_id": "plot_feature",
        "trigger": "Player selects a plot and confirms allocation",
        "intent": "Allocate plot", "effects": "Assign selected plot to player",
        "source_quote": "Allocate a plot", "runtime_state_reads": ["selected plot"],
    }
    symbols = {name: {"kind": "CLASS", "owner": name} for name in [
        "net.minecraft.commands.CommandSourceStack", "net.fabricmc.fabric.api.event.Event",
        "net.fabricmc.fabric.api.event.EventFactory",
        "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback"]}
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"id": "allocation", "entrypoints": {"main": ["example.Main"]}}))
    monkeypatch.setattr("minecraft_mod_ai.host_fabric_callback_abi.callback_interfaces", lambda *a, **kw: [])
    monkeypatch.setattr("minecraft_mod_ai.host_java_imports.host_add_verified_imports", lambda source, **kw: source)

    def body(_router, messages, *, data_name, frame, predicate, effect_receipt=False):
        if fail_body:
            raise ValueError("deliberate external model failure")
        return frame["prefix"] + "return true;" + frame["suffix"]
    monkeypatch.setattr("minecraft_mod_ai.host_callback_body.author_callback_body", body)

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            answers = {
                "source_owned_ui_activation": {"activation": "typed_ui_action", "trigger_quote": "Player selects a plot"},
                "ui_action_label": {"label": "Allocate plot"},
                "ui_action_input_count": {"count": 1},
                "ui_action_input_type": {"type": "string", "label": "Selected plot"},
            }
            return answers[data_name]
        def generate_java_source(self, *a, **kw):
            raise AssertionError("host owns the event, callback ABI, form and transport")

    graph = build_source_work_graph([behavior], producer_obligations=[{
        "behavior_id": "allocate", "name": "selected plot",
        "reason": "no_inspected_semantic_source",
    }], inspected_hooks=[])
    spec = {"language": "java", "target_path": "src/main/java/example/Feature.java",
            "requirement": "Allocate a plot", "side": "COMMON", "bindings": {
                "package_name": "example", "class_name": "Feature", "request_feature_entrypoint": True,
                "construction_behaviors_json": json.dumps([behavior])}}
    context = {"symbols": symbols, "host_source_owned": True, "source_owned_work_graph": graph,
               "authoring_contract": {}, "project_root": str(tmp_path),
               "target": {"minecraft_version": "26.2", "java_version": 25}}
    return Router(), behavior, spec, context, manifest


def test_missing_source_site_builds_a_real_user_action_with_server_actor(tmp_path, monkeypatch):
    from minecraft_mod_ai.source_owned_ui_event import read_verified_typed_ui_events
    router, behavior, spec, context, manifest = _fixture(tmp_path, monkeypatch)
    source = compose_first_pass_java(router, spec, [], construction_context=context, behaviors=[behavior])
    events = read_verified_typed_ui_events(project_root=tmp_path, symbols=context["symbols"],
                                         behaviors=[behavior], package="example")
    assert len(events) == 1
    assert events[0]["parameters"] == ["net.minecraft.commands.CommandSourceStack", "java.lang.String"]
    validate_registered_event_sources(source, [behavior], [], materialized_events=events)
    assert "source, input_1" in source
    entries = json.loads(manifest.read_text())["entrypoints"]
    assert len(entries["main"]) == 2 and len(entries["client"]) == 1


def test_failed_listener_authoring_does_not_install_the_planned_input_producer(tmp_path, monkeypatch):
    router, behavior, spec, context, manifest = _fixture(tmp_path, monkeypatch, fail_body=True)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="deliberate external model failure"):
        compose_first_pass_java(router, spec, [], construction_context=context, behaviors=[behavior])
    assert manifest.read_bytes() == original
    assert not (tmp_path / ".mmm/typed-ui-producers").exists()
    assert not list((tmp_path / "src").rglob("Occurrence.java"))


def test_full_source_compiler_stages_new_producer_dependencies(tmp_path, monkeypatch):
    from pathlib import Path

    from minecraft_mod_ai.construction_file_author import _verify_full_source
    from minecraft_mod_ai.construction_project_compile import _discover_closure

    router, behavior, spec, context, _ = _fixture(tmp_path, monkeypatch)
    source = compose_first_pass_java(router, spec, [], construction_context=context, behaviors=[behavior])
    context["java_classpath"] = ["/external-fixture"]
    staged = []
    def compiler(candidate, *, context, target_path, filename, source_context):
        staged.extend(str(row[0]) for row in _discover_closure(
            candidate, root=tmp_path, target=Path(target_path), source_context=source_context))
    monkeypatch.setattr("minecraft_mod_ai.construction_project_compile.verify_java_source_with_dependencies", compiler)
    _verify_full_source(source, spec, context, package="example", name="Feature")
    assert any(p.endswith("/Occurrence.java") for p in staged)
    assert any(p.endswith("/OccurrenceInitializer.java") for p in staged)
    assert any(p.endswith("/AuthoredActions.java") for p in staged)
    assert any(p.endswith("/AuthoredProgram.java") for p in staged)
    assert not any(p.startswith("src/client/") for p in staged)


def test_eligibility_trigger_and_action_effect_use_the_same_production_source_branch(tmp_path, monkeypatch):
    from minecraft_mod_ai.source_owned_ui_event import read_verified_typed_ui_events

    _, behavior, spec, context, _ = _fixture(tmp_path, monkeypatch)
    behavior.update({
        "trigger": "Player has enough resources and wants to expand the settlement",
        "effects": "Player commissions a new building for the settlement",
    })
    spec["bindings"]["construction_behaviors_json"] = json.dumps([behavior])
    context["source_owned_work_graph"] = build_source_work_graph(
        [behavior], producer_obligations=[], inspected_hooks=[])

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            return {
                "source_owned_player_action_evidence": {
                    "activation": "typed_ui_action", "request_field": "effects",
                    "request_quote": "Player commissions a new building for the settlement"},
                "ui_action_label": {"label": "Commission building"},
                "ui_action_input_count": {"count": 0},
            }[data_name]

    source = compose_first_pass_java(Router(), spec, [], construction_context=context,
                                     behaviors=[behavior])
    events = read_verified_typed_ui_events(project_root=tmp_path, symbols=context["symbols"],
                                         behaviors=[behavior], package="example")
    assert len(events) == 1
    validate_registered_event_sources(source, [behavior], [], materialized_events=events)
    assert events[0]["behavior_id"] == "allocate"


def test_mimo_unavailable_trade_still_materializes_a_real_original_action(tmp_path, monkeypatch):
    """Replay the source-owned branch that failed after ~2K production log lines."""
    from minecraft_mod_ai.source_owned_ui_event import read_verified_typed_ui_events

    _, behavior, spec, context, _ = _fixture(tmp_path, monkeypatch)
    behavior.update({
        "trigger": "Player has accumulated money and wants to build or upgrade a spacecraft",
        "intent": "Trade and purchase goods to fund spacecraft construction and upgrades",
        "effects": "Player trades mined resources for goods, then purchases spacecraft construction materials and upgrades",
        "source_quote": "거래 등으로",
    })
    spec["bindings"]["construction_behaviors_json"] = json.dumps([behavior])
    context["source_owned_work_graph"] = build_source_work_graph(
        [behavior], producer_obligations=[], inspected_hooks=[],
    )

    class DecliningPlanner:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            return {
                "source_owned_player_action_evidence": {
                    "activation": "unavailable", "request_field": "trigger", "request_quote": ""},
                "ui_action_label": {"label": "Trade goods"},
                "ui_action_input_count": {"count": 0},
            }[data_name]

    source = compose_first_pass_java(
        DecliningPlanner(), spec, [], construction_context=context, behaviors=[behavior],
    )
    events = read_verified_typed_ui_events(
        project_root=tmp_path, symbols=context["symbols"],
        behaviors=[behavior], package="example",
    )
    assert len(events) == 1
    assert events[0]["provenance"] == "host_typed_ui_action"
    assert events[0]["parameters"] == ["net.minecraft.commands.CommandSourceStack"]
    validate_registered_event_sources(source, [behavior], [], materialized_events=events)
