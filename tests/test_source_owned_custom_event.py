"""Real source-site event adapters: source/manifest integration, not a fake tick."""
import json
from pathlib import Path

import pytest

from minecraft_mod_ai.source_owned_custom_event import (
    plan_verified_custom_event, materialize_verified_custom_event,
)

SYMBOLS = {
    "opaqueOwner": {
        "kind": "CLASS", "owner": "example.Transaction",
        "name": "Transaction", "access": 1,
    },
    "opaqueMethod": {
        "kind": "METHOD", "owner": "example.Transaction",
        "name": "applyPurchase", "parameters": ["java.lang.String", "int"],
        "return_type": "boolean", "access": 1,
    },
}


def _plan(**kwargs):
    return plan_verified_custom_event(
        symbols=SYMBOLS, source_site_ref="opaqueMethod",
        behavior_id="buy_module", package="example.generated",
        mod_id="custommod", **kwargs,
    )


def test_verified_event_has_exact_observed_receiver_args_and_mixin_descriptor():
    result = _plan()
    event = result["callback"]
    assert event["parameters"] == ["example.Transaction", "java.lang.String", "int"]
    assert event["return_type"] == "void"
    assert event["provenance"] == "verified_source_site_mixin"
    files = result["files"]
    java = next(content for path, content in files.items() if path.endswith(".java")
                and "/mixin/" not in path)
    mixin = next(content for path, content in files.items() if "/mixin/" in path)
    assert "EventFactory.createArrayBacked" in java
    assert "void onEvent(example.Transaction v0, java.lang.String v1, int v2)" in java
    assert 'method = "applyPurchase(Ljava/lang/String;I)Z"' in mixin
    assert "CallbackInfoReturnable<?>" in mixin
    assert "arg0, arg1" in mixin
    assert "require = 1" in mixin
    assert result["runtime_verified"] is False
    assert json.loads(next(body for key, body in files.items()
                           if key.endswith(".mixins.json")))["compatibilityLevel"] == "JAVA_25"


def test_inspected_source_site_must_exist_and_be_injectable():
    with pytest.raises(ValueError, match="NOT_INSPECTED"):
        plan_verified_custom_event(
            symbols=SYMBOLS, source_site_ref="invented", behavior_id="buy",
            package="example.generated", mod_id="custommod")
    altered = {**SYMBOLS, "opaqueMethod": {**SYMBOLS["opaqueMethod"], "static": True}}
    with pytest.raises(ValueError, match="NOT_INJECTABLE"):
        plan_verified_custom_event(
            symbols=altered, source_site_ref="opaqueMethod",
            behavior_id="buy", package="example.generated", mod_id="custommod")
    missing_owner = {"opaqueMethod": SYMBOLS["opaqueMethod"]}
    with pytest.raises(ValueError, match="OWNER_NOT_INSPECTED"):
        plan_verified_custom_event(
            symbols=missing_owner, source_site_ref="opaqueMethod",
            behavior_id="buy", package="example.generated", mod_id="custommod")
    with pytest.raises(ValueError, match="INJECTION_POINT_INVALID"):
        _plan(at="ARBITRARY")


def test_materialization_is_idempotent_preserves_existing_mixins(tmp_path: Path):
    path = tmp_path / "src/main/resources/fabric.mod.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schemaVersion": 1, "id": "custommod",
                                "mixins": ["existing.mixins.json"]}))
    options = dict(
        project_root=tmp_path, symbols=SYMBOLS,
        source_site_ref="opaqueMethod", behavior_id="buy_module",
        package="example.generated",
    )
    first = materialize_verified_custom_event(**options)
    second = materialize_verified_custom_event(**options)
    assert first == second
    manifest = json.loads(path.read_text())
    assert manifest["mixins"] == ["existing.mixins.json", first["mixin_config"]]
    assert all((tmp_path / file).read_text() == source
               for file, source in first["files"].items())
    assert json.loads((tmp_path / ("src/main/resources/" + first["mixin_config"]))
                      .read_text())["required"] is True
    event_path = next(tmp_path / key for key in first["files"] if key.endswith(".java")
                      and "/mixin/" not in key)
    event_path.write_text("malicious or stale bytes")
    with pytest.raises(ValueError, match="EXISTING_FILE_CONFLICT"):
        materialize_verified_custom_event(**options)


def test_generated_event_java_compiles_and_dispatches_observed_arguments(tmp_path):
    import shutil
    import subprocess
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")
    result = _plan()
    event_name = result["callback"]["event_owner"].rsplit(".", 1)[-1]
    stubs = {
        "example/Transaction.java": "package example; public class Transaction { public boolean applyPurchase(String name, int count) { return true; }}",
        "org/spongepowered/asm/mixin/Mixin.java":
            "package org.spongepowered.asm.mixin; public @interface Mixin { Class<?> value(); }",
        "org/spongepowered/asm/mixin/injection/At.java":
            'package org.spongepowered.asm.mixin.injection; public @interface At { String value(); }',
        "org/spongepowered/asm/mixin/injection/Inject.java":
            "package org.spongepowered.asm.mixin.injection; public @interface Inject { String method(); At at(); int require() default 1; }",
        "org/spongepowered/asm/mixin/injection/callback/CallbackInfoReturnable.java":
            "package org.spongepowered.asm.mixin.injection.callback; public class CallbackInfoReturnable<T> {}",
        "net/fabricmc/fabric/api/event/Event.java": """package net.fabricmc.fabric.api.event;
public class Event<T> {
 private final Class<T> type; private final java.util.function.Function<T[],T> factory;
 private final java.util.List<T> members = new java.util.ArrayList<>();
 public Event(Class<T> t, java.util.function.Function<T[],T> f) { type=t; factory=f; }
 public void register(T member) { members.add(member); }
 @SuppressWarnings("unchecked") public T invoker() {
  T[] arr=(T[])java.lang.reflect.Array.newInstance(type,members.size());
  return factory.apply(members.toArray(arr));
 }
}""",
        "net/fabricmc/fabric/api/event/EventFactory.java": """package net.fabricmc.fabric.api.event;
public class EventFactory {
 public static <T> Event<T> createArrayBacked(Class<T> type, java.util.function.Function<T[],T> fn) {
  return new Event<>(type,fn);
 }
}""",
        "example/generated/Harness.java": f"""package example.generated;
public class Harness {{
 public static void main(String[] args) {{
  int[] seen = new int[] {{0}};
  {event_name}.EVENT.register((tx, name, count) -> {{
   if (tx == null || !name.equals("fuel") || count != 3) throw new AssertionError("lost argument");
   seen[0]++;
  }});
  {event_name}.EVENT.invoker().onEvent(new example.Transaction(), "fuel", 3);
  if (seen[0] != 1) throw new AssertionError("not dispatched");
 }}
}}""",
    }
    files = dict(stubs)
    files.update({name[len("src/main/java/"):]: body for name, body in result["files"].items()
                  if name.startswith("src/main/java/")})
    sources = []
    for name, body in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        sources.append(str(target))
    finished = subprocess.run([javac, "-d", str(tmp_path), *sources],
                              capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    run = subprocess.run([java, "-cp", str(tmp_path), "example.generated.Harness"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_source_required_materializes_mixin_and_binds_real_generated_event(tmp_path, monkeypatch):
    from minecraft_mod_ai.feature_source_work_graph import build_source_work_graph
    from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java

    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 1, "id": "custommod"}))
    observed_callback = {
        "event_id": "fabric#existing", "event_owner": "fabric",
        "event_field": "EXISTING", "event_interface": "fabric.Listener",
        "method": "observe", "parameters": ["example.Transaction"],
        "return_type": "void", "type_parameters": [], "throws": [],
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.host_fabric_callback_abi.callback_interfaces",
        lambda *_args, **_kwargs: [observed_callback],
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.host_java_imports.host_add_verified_imports",
        lambda source, **_kw: source,
    )
    implemented = []
    def fake_body(_router, _messages, *, data_name, frame, predicate,
                  effect_receipt=False):
        implemented.append((data_name, predicate, effect_receipt))
        body = "return true;" if predicate or effect_receipt else ""
        return frame["prefix"] + body + frame["suffix"]
    monkeypatch.setattr(
        "minecraft_mod_ai.host_callback_body.author_callback_body",
        fake_body,
    )

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            assert data_name == "select_source_owned_observation_hook"
            assert role == "planner"
            return {"event_id": "SOURCE_REQUIRED"}
        def generate_java_source(self, *_args, **_kwargs):
            raise AssertionError("host must own source and Java callback envelope")

    behavior = {
        "behavior_id": "buy_module", "intent": "Apply an inventory purchase",
        "trigger": "Observed successful transaction method invocation",
        "effects": "Apply purchased module to player state",
        "source_site_ref": "opaqueMethod",
    }
    graph = build_source_work_graph(
        [behavior], producer_obligations=[
            {"behavior_id": "buy_module", "reason": "occurrence_hook_not_owned"},
        ],
        inspected_hooks=[observed_callback],
    )
    spec = {
        "language": "java", "target_path": "src/main/java/example/generated/Bridge.java",
        "requirement": "Execute the original transaction feature",
        "bindings": {
            "package_name": "example.generated", "class_name": "Bridge",
            "request_feature_entrypoint": True,
        },
    }
    source = compose_first_pass_java(
        Router(), spec, [], behaviors=[behavior],
        construction_context={
            "symbols": SYMBOLS, "host_source_owned": True,
            "source_owned_work_graph": graph, "project_root": str(tmp_path),
            "authoring_contract": {},
        },
    )
    manifest_json = json.loads(manifest.read_text())
    assert len(manifest_json["mixins"]) == 1
    assert ".EVENT.register(this::handleBehavior" in source
    assert "fabric.EXISTING" not in source
    assert "example.generated.MmmEvent" in source
    assert "private boolean matchesBehavior" in source
    assert len(implemented) == 2



def test_client_mixin_uses_client_source_set_and_scoped_manifest(tmp_path):
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    existing = {"config": "another.client.mixins.json", "environment": "client"}
    manifest.write_text(json.dumps({
        "schemaVersion": 1, "id": "custommod", "mixins": [existing],
    }))
    created = materialize_verified_custom_event(
        project_root=tmp_path, symbols=SYMBOLS,
        source_site_ref="opaqueMethod", behavior_id="client_purchase",
        package="example.generated", side="CLIENT", java_version=25,
    )
    files = created["files"]
    assert any(name.startswith("src/client/java/") for name in files)
    assert not any(name.startswith("src/main/java/") for name in files)
    mixin = json.loads(files["src/main/resources/" + created["mixin_config"]])
    assert mixin["compatibilityLevel"] == "JAVA_25"
    assert mixin["client"] == [next(name.rsplit("/", 1)[-1].removesuffix(".java")
                                    for name in files if "/mixin/" in name)]
    mixins = json.loads(manifest.read_text())["mixins"]
    assert mixins == [existing, {
        "config": created["mixin_config"], "environment": "client",
    }]



def test_materialized_custom_event_requires_real_byte_verified_registration(tmp_path):
    import hashlib
    from minecraft_mod_ai.feature_registration_proof import validate_registered_event_sources
    from minecraft_mod_ai.source_owned_custom_event import read_verified_custom_event_receipts

    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 1, "id": "custommod"}))
    installed = materialize_verified_custom_event(
        project_root=tmp_path, symbols=SYMBOLS,
        source_site_ref="opaqueMethod", behavior_id="buy_module",
        package="example.generated",
    )
    receipts = read_verified_custom_event_receipts(
        project_root=tmp_path, symbols=SYMBOLS,
        behavior_ids=["buy_module"], package="example.generated",
    )
    assert len(receipts) == 1
    assert receipts[0] == installed["callback"]
    suffix = hashlib.sha256(b"buy_module").hexdigest()[:20]
    event_owner = receipts[0]["event_owner"]
    java = (f"public class Bridge {{\n"
            f"private void handleBehavior{suffix}(example.Transaction tx, String name, int count) {{ }}\n"
            f"private void registerBehavior{suffix}() {{"
            f"{event_owner}.EVENT.register(this::handleBehavior{suffix}); }}\n"
            "}")
    validate_registered_event_sources(
        java, [{"behavior_id": "buy_module"}], [],
        materialized_events=receipts,
    )
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(
            java, [{"behavior_id": "buy_module"}], [],
        )
    producer = next(tmp_path / path for path in installed["files"]
                    if path.endswith(".java") and "/mixin/" in path)
    producer.write_text("tampered mixin source")
    with pytest.raises(ValueError, match="MATERIALIZED_FILE_MISMATCH"):
        read_verified_custom_event_receipts(
            project_root=tmp_path, symbols=SYMBOLS,
            behavior_ids=["buy_module"], package="example.generated",
        )


def test_source_publication_conflicting_receipt_has_no_side_effects(tmp_path):
    import hashlib
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    original = json.dumps({"schemaVersion": 1, "id": "custommod", "mixins": []})
    manifest.write_text(original)
    digest = hashlib.sha256(b"buy_module\0opaqueMethod\0HEAD").hexdigest()[:16]
    receipt = tmp_path / ".mmm/source-producers" / (digest + ".json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("conflict")
    with pytest.raises(ValueError, match="SOURCE_SITE_RECEIPT_CONFLICT"):
        materialize_verified_custom_event(
            project_root=tmp_path, symbols=SYMBOLS, source_site_ref="opaqueMethod",
            behavior_id="buy_module", package="example.generated")
    assert manifest.read_text() == original
    assert receipt.read_text() == "conflict"
    assert all(not (tmp_path / name).exists() for name in _plan()["files"])


@pytest.mark.parametrize("fail_path", ["manifest", "receipt"])
def test_source_publication_rolls_back_post_replace_failure(tmp_path, monkeypatch, fail_path):
    import minecraft_mod_ai.source_owned_custom_event as publisher
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    original = json.dumps({"schemaVersion": 1, "id": "custommod", "mixins": []})
    manifest.write_text(original)
    write = publisher._atomic_write
    failed = []
    def interrupted(path, contents):
        write(path, contents)
        category = ("manifest" if path == manifest else
                    "receipt" if "source-producers" in str(path) else "source")
        if category == fail_path and not failed:
            failed.append(category)
            raise RuntimeError("post-replace publication failure")
    monkeypatch.setattr(publisher, "_atomic_write", interrupted)
    with pytest.raises(RuntimeError, match="post-replace publication failure"):
        materialize_verified_custom_event(
            project_root=tmp_path, symbols=SYMBOLS, source_site_ref="opaqueMethod",
            behavior_id="buy_module", package="example.generated")
    assert failed == [fail_path]
    assert manifest.read_text() == original
    assert all(not (tmp_path / name).exists() for name in _plan()["files"])
    assert not list((tmp_path / ".mmm/source-producers").glob("*.json"))


def test_target_compile_includes_verified_injection_class_not_just_event_owner(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import read_verified_custom_event_receipts
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 1, "id": "custommod"}))
    built = materialize_verified_custom_event(
        project_root=tmp_path, symbols=SYMBOLS,
        source_site_ref="opaqueMethod", behavior_id="buy_module",
        package="example.generated")
    verified = read_verified_custom_event_receipts(
        project_root=tmp_path, symbols=SYMBOLS,
        behavior_ids=["buy_module"], package="example.generated",
        include_producer_paths=True)
    assert len(verified) == 1
    expected = sorted(path for path in built["files"] if path.endswith(".java"))
    assert verified[0]["producer_source_paths"] == expected
    assert len(expected) == 2
    assert any("/mixin/" in path for path in expected)
    original = read_verified_custom_event_receipts(
        project_root=tmp_path, symbols=SYMBOLS,
        behavior_ids=["buy_module"], package="example.generated")
    assert original == [built["callback"]]


def test_byte_identical_symlink_cannot_impersonate_verified_mixin_producer(tmp_path):
    from minecraft_mod_ai.source_owned_custom_event import read_verified_custom_event_receipts
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schemaVersion": 1, "id": "custommod"}))
    options = {
        "project_root": tmp_path, "symbols": SYMBOLS,
        "source_site_ref": "opaqueMethod", "behavior_id": "buy_module",
        "package": "example.generated",
    }
    result = materialize_verified_custom_event(**options)
    java_path = tmp_path / next(path for path in result["files"] if "/mixin/" in path)
    other = tmp_path / "same-byte-target.java"
    other.write_bytes(java_path.read_bytes())
    java_path.unlink()
    try:
        java_path.symlink_to(other)
    except OSError:
        pytest.skip("symlink creation is unavailable on this filesystem")
    with pytest.raises(ValueError, match="SOURCE_SITE_PATH_UNSAFE"):
        read_verified_custom_event_receipts(
            project_root=tmp_path, symbols=SYMBOLS,
            behavior_ids=["buy_module"], package="example.generated")
    with pytest.raises(ValueError, match="SOURCE_SITE_PATH_UNSAFE"):
        materialize_verified_custom_event(**options)
