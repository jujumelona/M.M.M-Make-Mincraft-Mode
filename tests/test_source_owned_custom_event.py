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
