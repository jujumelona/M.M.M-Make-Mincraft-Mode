"""A requested action must have a real input producer, not a fabricated hook."""
import json
import shutil
import subprocess

import pytest

BEHAVIOR = {
    "behavior_id": "allocate_plot", "requirement_id": "allocate",
    "trigger": "Player selects a plot and confirms allocation",
    "intent": "Allocate the selected plot", "effects": "Assign the plot to the player",
    "source_quote": "allocate a plot",
}
METADATA = {"intent": "Allocate plot", "inputs": [
    {"name": "plot", "type": "string", "label": "Plot", "max_length": 32,
     "choices": ["north", "south"]},
    {"name": "count", "type": "int", "label": "Count", "minimum": 1, "maximum": 3},
]}
SYMBOLS = {name: {"kind": "CLASS", "owner": name, "name": name.rsplit(".", 1)[-1]}
           for name in ["net.minecraft.commands.CommandSourceStack",
                        "net.fabricmc.fabric.api.event.Event",
                        "net.fabricmc.fabric.api.event.EventFactory",
                        "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback"]}


def _plan(**changes):
    from minecraft_mod_ai.source_owned_ui_event import plan_typed_ui_occurrence
    return plan_typed_ui_occurrence(
        behavior=BEHAVIOR, action_metadata=METADATA, package="example.generated",
        mod_id="allocation", minecraft_version="26.2", symbols=SYMBOLS, **changes,
    )


def test_validated_player_inputs_reach_real_listener_and_declined_effect_returns_zero(tmp_path):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required for generated Java execution")
    result = _plan()
    event = result["callback"]
    package = event["event_owner"].rpartition(".")[0]
    assert event["parameters"] == ["net.minecraft.commands.CommandSourceStack", "java.lang.String", "int"]
    action = result["action"]["id"]
    stubs = {
        "net/minecraft/commands/CommandSourceStack.java": "package net.minecraft.commands; public class CommandSourceStack {}",
        "net/fabricmc/fabric/api/event/Event.java": """package net.fabricmc.fabric.api.event;
public final class Event<T> {
 private final Class<T> type; private final java.util.function.Function<T[],T> factory;
 private final java.util.List<T> listeners = new java.util.ArrayList<>();
 public Event(Class<T> t, java.util.function.Function<T[],T> f) { type=t; factory=f; }
 public void register(T listener) { listeners.add(listener); }
 @SuppressWarnings("unchecked") public T invoker() {
  return factory.apply(listeners.toArray((T[])java.lang.reflect.Array.newInstance(type,0)));
 }
}""",
        "net/fabricmc/fabric/api/event/EventFactory.java": """package net.fabricmc.fabric.api.event;
public final class EventFactory {
 public static <T> Event<T> createArrayBacked(Class<T> type, java.util.function.Function<T[],T> fn) {
  return new Event<>(type,fn);
 }
}""",
        package.replace(".", "/") + "/Harness.java": f"""package {package};
public final class Harness {{
 public static void main(String[] args) {{
  var actor = new net.minecraft.commands.CommandSourceStack();
  if (AuthoredActions.invoke("{action}", actor, java.util.Map.of("plot","north","count",2)) != 0)
   throw new AssertionError("no listener must decline");
  int[] allocated = {{0}};
  Occurrence.EVENT.register((source, plot, count) -> {{
   if (source != actor || !plot.equals("north")) throw new AssertionError("lost actor/input");
   if (count == 3) return false;
   allocated[0] += count; return true;
  }});
  String command = AuthoredActions.commandFromText("{action}", java.util.Map.of("plot","north","count","2"));
  var tokens = java.util.Arrays.asList(command.split(" ")).subList(1,3);
  if (AuthoredActions.submitEncoded("{action}", actor, tokens) != 1 || allocated[0] != 2)
   throw new AssertionError("validated server action did not apply");
  if (AuthoredActions.invoke("{action}", actor, java.util.Map.of("plot","north","count",3)) != 0 || allocated[0] != 2)
   throw new AssertionError("declined effect was reported successful");
  for (var invalid : java.util.List.<java.util.Map<String,Object>>of(java.util.Map.of("plot","forged","count",2),
       java.util.Map.of("plot","north","count",4), java.util.Map.of("plot","north","count",2,"credits",100))) {{
   try {{ AuthoredActions.invoke("{action}", actor, invalid); throw new AssertionError("accepted forged inputs"); }}
   catch (IllegalArgumentException expected) {{}}
  }}
  if (allocated[0] != 2) throw new AssertionError("rejected input mutated state");
 }}
}}""",
    }
    for path, source in result["files"].items():
        if path.startswith("src/main/java/") and not path.endswith("OccurrenceInitializer.java"):
            stubs[path.removeprefix("src/main/java/")] = source
    files = []
    for name, content in stubs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        files.append(str(path))
    compiled = subprocess.run([javac, "-d", str(tmp_path / "classes"), *files], capture_output=True, text=True, timeout=60, check=False)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([java, "-cp", str(tmp_path / "classes"), package + ".Harness"], capture_output=True, text=True, timeout=30, check=False)
    assert executed.returncode == 0, executed.stderr


def test_install_preserves_manifest_and_replay_requires_exact_owned_bytes(tmp_path):
    from minecraft_mod_ai.source_owned_ui_event import (
        materialize_typed_ui_occurrence,
        read_verified_typed_ui_events,
    )
    result = _plan()
    path = tmp_path / "src/main/resources/fabric.mod.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"id": "allocation", "entrypoints": {
        "main": ["example.Existing"], "client": [{"value": "example.Client"}]},
        "mixins": ["existing.mixins.json"]}))
    materialize_typed_ui_occurrence(result, project_root=tmp_path, symbols=SYMBOLS)
    materialize_typed_ui_occurrence(result, project_root=tmp_path, symbols=SYMBOLS)
    manifest = json.loads(path.read_text())
    assert manifest["entrypoints"]["main"][0] == "example.Existing"
    assert len(manifest["entrypoints"]["main"]) == 2
    assert manifest["entrypoints"]["client"][0] == {"value": "example.Client"}
    assert manifest["mixins"] == ["existing.mixins.json"]
    options = {"project_root": tmp_path, "symbols": SYMBOLS,
               "behaviors": [BEHAVIOR], "package": "example.generated"}
    assert read_verified_typed_ui_events(**options) == [result["callback"]]
    producer = tmp_path / next(k for k in result["files"] if k.endswith("/Occurrence.java"))
    producer.write_text(producer.read_text() + "// changed\n")
    with pytest.raises(ValueError, match="MATERIALIZED_FILE_MISMATCH"):
        read_verified_typed_ui_events(**options)


def test_source_planning_has_no_file_side_effects_and_passive_triggers_cannot_become_forms(tmp_path):
    from minecraft_mod_ai.source_owned_hook import SourceHookUnavailable
    from minecraft_mod_ai.source_owned_ui_event import author_typed_ui_occurrence
    class Router:
        def generate_structured_data(self, *args, **kwargs):
            return {"activation": "typed_ui_action", "trigger_quote": "Player reaches a new world"}
    passive = {**BEHAVIOR, "trigger": "Player reaches a new world"}
    with pytest.raises(SourceHookUnavailable):
        author_typed_ui_occurrence(Router(), passive, package="example.generated", mod_id="allocation",
                                   minecraft_version="26.2", symbols=SYMBOLS)
    assert list(tmp_path.iterdir()) == []


def test_automatic_server_selection_is_not_a_player_input():
    from minecraft_mod_ai.source_owned_hook import SourceHookUnavailable
    from minecraft_mod_ai.source_owned_ui_event import plan_typed_ui_occurrence
    with pytest.raises(SourceHookUnavailable):
        plan_typed_ui_occurrence(
            behavior={**BEHAVIOR, "trigger": "Server selects a new target automatically"},
            action_metadata=METADATA, package="example.generated", mod_id="allocation",
            minecraft_version="26.2", symbols=SYMBOLS,
        )


def test_install_rejects_modified_plan_before_publishing_files(tmp_path):
    from minecraft_mod_ai.source_owned_ui_event import materialize_typed_ui_occurrence
    planned = _plan()
    path = tmp_path / "src/main/resources/fabric.mod.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"allocation"}')
    original = path.read_bytes()
    planned["files"]["src/main/java/example/Injected.java"] = "unowned source"
    with pytest.raises(ValueError, match="PLAN_MISMATCH"):
        materialize_typed_ui_occurrence(planned, project_root=tmp_path, symbols=SYMBOLS)
    assert path.read_bytes() == original
    assert not (tmp_path / "src/main/java").exists()


@pytest.mark.parametrize("trigger", [
    "Player enters a region and the server selects a target automatically",
    "Player reaches the destination without having to click or confirm anything",
    "Player does not select any target",
    "Player selects a target automatically",
])
def test_passive_and_negated_clauses_are_not_ui_activation_even_if_model_claims_otherwise(trigger):
    from minecraft_mod_ai.source_owned_hook import SourceHookUnavailable
    from minecraft_mod_ai.source_owned_ui_event import author_typed_ui_occurrence
    class Router:
        def generate_structured_data(self, *args, data_name, **kwargs):
            return {"source_owned_ui_activation": {"activation": "typed_ui_action", "trigger_quote": trigger},
                    "ui_action_label": {"label": "Allocate plot"},
                    "ui_action_input_count": {"count": 0}}[data_name]
    with pytest.raises(SourceHookUnavailable):
        author_typed_ui_occurrence(
            Router(), {**BEHAVIOR, "trigger": trigger}, symbols=SYMBOLS,
            package="example.generated", mod_id="allocation", minecraft_version="26.2")


def test_two_actions_cannot_both_be_proved_by_registering_only_one_producer():
    import hashlib

    from minecraft_mod_ai.feature_registration_proof import (
        validate_registered_event_sources,
    )
    from minecraft_mod_ai.source_owned_ui_event import plan_typed_ui_occurrence
    allocate = _plan()
    release_behavior = {**BEHAVIOR, "behavior_id": "release_plot", "trigger": "Player confirms release"}
    release = plan_typed_ui_occurrence(
        behavior=release_behavior, action_metadata=METADATA, package="example.generated",
        mod_id="allocation", minecraft_version="26.2", symbols=SYMBOLS)
    methods = []
    for behavior in [BEHAVIOR, release_behavior]:
        suffix = hashlib.sha256(behavior["behavior_id"].encode()).hexdigest()[:20]
        methods.append(f'private void registerBehavior{suffix}() {{ '
                       f'{allocate["callback"]["event_owner"]}.EVENT.register(this::handleBehavior{suffix}); }}')
    source = 'class Feature { ' + ' '.join(methods) + ' }'
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(source, [BEHAVIOR, release_behavior], [],
                                          materialized_events=[allocate["callback"], release["callback"]])


@pytest.mark.parametrize("change", ["metadata", "behavior"])
def test_reauthoring_replaces_the_previous_owned_producer_without_duplicate_startup(tmp_path, change):
    from minecraft_mod_ai.source_owned_ui_event import (
        materialize_typed_ui_occurrence,
        plan_typed_ui_occurrence,
        read_verified_typed_ui_events,
    )
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"allocation","entrypoints":{"main":["example.Main"]}}')
    first = _plan()
    materialize_typed_ui_occurrence(first, project_root=tmp_path, symbols=SYMBOLS)
    behavior = {**BEHAVIOR, **({"trigger": "Player confirms allocation of selected plot"} if change == "behavior" else {})}
    metadata = {**METADATA, "intent": "Allocate the plot"}
    second = plan_typed_ui_occurrence(
        behavior=behavior, action_metadata=metadata, package="example.generated",
        mod_id="allocation", minecraft_version="26.2", symbols=SYMBOLS)
    materialize_typed_ui_occurrence(second, project_root=tmp_path, symbols=SYMBOLS)
    assert read_verified_typed_ui_events(project_root=tmp_path, symbols=SYMBOLS,
        behaviors=[behavior], package="example.generated") == [second["callback"]]
    entries = json.loads(manifest.read_text())["entrypoints"]
    assert entries["main"] == ["example.Main", second["entrypoints"]["main"]]
    assert entries["client"] == [second["entrypoints"]["client"]]
    assert not any((tmp_path / name).exists() for name in first["files"])
    assert len(list((tmp_path / ".mmm/typed-ui-producers").glob("*.json"))) == 1


def test_reauthoring_preserves_user_modified_old_producer(tmp_path):
    from minecraft_mod_ai.source_owned_ui_event import (
        materialize_typed_ui_occurrence,
        plan_typed_ui_occurrence,
    )
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"allocation"}')
    first = _plan()
    materialize_typed_ui_occurrence(first, project_root=tmp_path, symbols=SYMBOLS)
    changed = tmp_path / next(k for k in first["files"] if k.endswith("/Occurrence.java"))
    changed.write_text("user source")
    original = manifest.read_bytes()
    second = plan_typed_ui_occurrence(
        behavior=BEHAVIOR, action_metadata={**METADATA, "intent": "New label"},
        package="example.generated", mod_id="allocation", minecraft_version="26.2", symbols=SYMBOLS)
    with pytest.raises(ValueError, match="CONFLICT"):
        materialize_typed_ui_occurrence(second, project_root=tmp_path, symbols=SYMBOLS)
    assert changed.read_text() == "user source"
    assert manifest.read_bytes() == original
    assert not any((tmp_path / name).exists() for name in second["files"])


def test_ui_install_waits_for_the_existing_mixin_manifest_writer(tmp_path):
    import fcntl
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from minecraft_mod_ai.source_owned_ui_event import materialize_typed_ui_occurrence
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"allocation"}')
    shared_lock = tmp_path / ".mmm/source-producer.lock"
    shared_lock.parent.mkdir(parents=True)
    started = Event()
    def install():
        started.set()
        materialize_typed_ui_occurrence(_plan(), project_root=tmp_path, symbols=SYMBOLS)
    with shared_lock.open("a+") as owner, ThreadPoolExecutor(max_workers=1) as executor:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
        future = executor.submit(install)
        assert started.wait(2)
        try:
            with pytest.raises(TimeoutError):
                future.result(timeout=0.5)
            assert json.loads(manifest.read_text()) == {"id": "allocation"}
        finally:
            fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
        future.result(timeout=5)


@pytest.mark.parametrize("producer", ["ui", "mixin"])
def test_occurrence_install_waits_for_project_entrypoint_reconciliation(tmp_path, producer):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from minecraft_mod_ai.project_write_lock import project_write_lock
    from minecraft_mod_ai.source_owned_custom_event import (
        materialize_verified_custom_event,
    )
    from minecraft_mod_ai.source_owned_ui_event import materialize_typed_ui_occurrence
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"allocation"}')
    started = Event()
    def install():
        started.set()
        if producer == "ui":
            return materialize_typed_ui_occurrence(_plan(), project_root=tmp_path, symbols=SYMBOLS)
        return materialize_verified_custom_event(
            project_root=tmp_path, symbols={
                "target": {"kind": "CLASS", "owner": "example.Transaction"},
                "apply": {"kind": "METHOD", "owner": "example.Transaction", "name": "apply",
                          "parameters": [], "return_type": "void", "access": 1},
            }, source_site_ref="apply", behavior_id="allocate_plot", package="example.generated")
    with ThreadPoolExecutor(max_workers=1) as executor:
        with project_write_lock(tmp_path):
            future = executor.submit(install)
            assert started.wait(2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.5)
            assert json.loads(manifest.read_text()) == {"id": "allocation"}
        future.result(timeout=5)
