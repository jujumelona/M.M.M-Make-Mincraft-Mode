"""All validated Java artifacts, not only one prompt, reach actual Fabric startup."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.host_fabric_initializer_wiring import wire_completed_fabric_java
from minecraft_mod_ai.artifact_graph_executor import _wire_validated_java_startup


def _write(root: Path, relative: str, value: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(root):
    _write(root, "src/main/resources/fabric.mod.json", json.dumps({
        "id": "example", "entrypoints": {"main": ["proof.ExistingInitializer"]}
    }))
    registry = "src/main/java/proof/generated/Things.java"
    feature = "src/main/java/proof/actions/Feature.java"
    client = "src/main/java/proof/actions/ClientFeature.java"
    _write(root, registry, """package proof.generated;
public final class Things {
    public static int runs = 0;
    public static void register() { runs++; }
}""")
    _write(root, feature, """package proof.actions;
import net.fabricmc.api.ModInitializer;
public final class Feature implements ModInitializer {
    @Override public void onInitialize() { proof.generated.Things.runs += 10; }
}""")
    _write(root, client, """package proof.actions;
public final class ClientFeature implements net.fabricmc.api.ClientModInitializer {
    @Override public void onInitializeClient() { proof.generated.Things.runs += 100; }
}""")
    return registry, feature, client


def test_startup_entries_are_derived_from_admitted_java_abi_and_idempotent(tmp_path):
    registry, feature, client = _fixture(tmp_path)
    approved = [(registry, "minecraft/entity/registry"),
                (feature, "minecraft/event/callback"),
                (client, "minecraft/client/event")]
    result = wire_completed_fabric_java(tmp_path, approved)
    assert result["status"] == "PASS"
    metadata_file = tmp_path / "src/main/resources/fabric.mod.json"
    metadata = json.loads(metadata_file.read_text())
    main = metadata["entrypoints"]["main"]
    assert main[0] == "proof.ExistingInitializer"
    assert "proof.actions.Feature" in main
    assert "proof.actions.ClientFeature" in metadata["entrypoints"]["client"]
    registrars = [e for e in main if e.startswith("proof.generated.MmmRegistryBootstrap")]
    assert len(registrars) == 1
    java_path = tmp_path / "src/main/java/proof/generated" / (
        registrars[0].rsplit(".", 1)[-1] + ".java"
    )
    generated = java_path.read_text()
    assert "proof.generated.Things.register();" in generated
    assert "onInitialize()" in generated
    previous = metadata_file.read_bytes()
    wire_completed_fabric_java(tmp_path, approved)
    assert metadata_file.read_bytes() == previous
    assert java_path.read_text() == generated


def test_javac_compiles_and_invokes_generated_registry_bootstrap(tmp_path):
    javac = shutil.which("javac")
    if javac is None:
        pytest.skip("JDK not installed")
    registry, feature, client = _fixture(tmp_path)
    wire_completed_fabric_java(
        tmp_path, [(registry, "minecraft/entity/registry"),
                   (feature, "minecraft/event/callback"),
                   (client, "minecraft/client/event")],
    )
    metadata = json.loads(
        (tmp_path / "src/main/resources/fabric.mod.json").read_text())
    bootstrap = next(
        entry for entry in metadata["entrypoints"]["main"]
        if "MmmRegistryBootstrap" in entry
    )
    _write(tmp_path, "src/main/java/net/fabricmc/api/ModInitializer.java",
           "package net.fabricmc.api; public interface ModInitializer { void onInitialize(); }")
    _write(tmp_path, "src/main/java/net/fabricmc/api/ClientModInitializer.java",
           "package net.fabricmc.api; public interface ClientModInitializer { void onInitializeClient(); }")
    _write(tmp_path, "src/main/java/proof/Probe.java", """package proof;
public final class Probe {
    public static void main(String[] args) throws Exception {
        Class<?> kind = Class.forName(args[0]);
        Object instance = kind.getDeclaredConstructor().newInstance();
        ((net.fabricmc.api.ModInitializer) instance).onInitialize();
        if (proof.generated.Things.runs != 1) throw new AssertionError("register did not run");
        new proof.actions.Feature().onInitialize();
        if (proof.generated.Things.runs != 11) throw new AssertionError("feature did not run");
        new proof.actions.ClientFeature().onInitializeClient();
        if (proof.generated.Things.runs != 111) throw new AssertionError("client did not run");
    }
}""")
    sources = [str(path) for path in (tmp_path / "src/main/java").rglob("*.java")]
    dest = tmp_path / "classes"
    dest.mkdir()
    compilation = subprocess.run(
        [javac, "-proc:none", "-d", str(dest), *sources],
        capture_output=True, text=True, timeout=80,
    )
    assert compilation.returncode == 0, compilation.stderr
    runtime = subprocess.run(
        [str(Path(javac).with_name("java")), "-cp", str(dest),
         "proof.Probe", bootstrap],
        capture_output=True, text=True, timeout=20,
    )
    assert runtime.returncode == 0, runtime.stderr


def test_graph_wires_only_materialized_validated_receipt(tmp_path):
    registry, _, _ = _fixture(tmp_path)
    job = SimpleNamespace(canonical_leaf="minecraft/entity/registry")
    import hashlib
    receipt = {"status": "PASS", "materialization": {
        "path": str(tmp_path / registry),
        "after_sha256": hashlib.sha256((tmp_path / registry).read_bytes()).hexdigest(),
    }}
    _wire_validated_java_startup(
        [job], [receipt], context={"project_root": str(tmp_path)}, base_dir=tmp_path,
    )
    metadata = json.loads(
        (tmp_path / "src/main/resources/fabric.mod.json").read_text())
    assert any("MmmRegistryBootstrap" in x
               for x in metadata["entrypoints"]["main"])


def test_no_registry_invocation_is_invented_when_method_missing(tmp_path):
    registry, _, _ = _fixture(tmp_path)
    java = tmp_path / registry
    java.write_text(java.read_text().replace("public static void register()", "public void register()"))
    with pytest.raises(ValueError, match="FABRIC_INIT_REGISTRY_ENTRYPOINT_MISSING"):
        wire_completed_fabric_java(tmp_path, [(registry, "minecraft/entity/registry")])


def test_non_sourced_file_and_symlink_cannot_be_declared_verified(tmp_path):
    registry, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="FABRIC_INIT_JAVA_PATH_UNVERIFIED"):
        wire_completed_fabric_java(tmp_path, [("../outside.java", "minecraft/entity/registry")])
    real = tmp_path / registry
    real.rename(real.with_name("Actual.java"))
    real.symlink_to(real.with_name("Actual.java"))
    with pytest.raises(ValueError, match="FABRIC_INIT_JAVA_SYMLINK"):
        wire_completed_fabric_java(tmp_path, [(registry, "minecraft/entity/registry")])



def test_source_change_after_graph_receipt_must_not_bind_startup(tmp_path):
    import hashlib
    from minecraft_mod_ai.artifact_graph_executor import ArtifactGraphError
    registry, _, _ = _fixture(tmp_path)
    java_path = tmp_path / registry
    stamped = hashlib.sha256(java_path.read_bytes()).hexdigest()
    java_path.write_text(java_path.read_text() + "\n// unrelated edit")
    job = SimpleNamespace(canonical_leaf="minecraft/entity/registry")
    receipt = {"status": "PASS", "materialization": {
        "path": str(java_path), "after_sha256": stamped,
    }}
    with pytest.raises(ArtifactGraphError, match="ARTIFACT_JAVA_STARTUP_SOURCE_UNVERIFIED"):
        _wire_validated_java_startup(
            [job], [receipt], context={"project_root": str(tmp_path)}, base_dir=tmp_path,
        )
    metadata = json.loads(
        (tmp_path / "src/main/resources/fabric.mod.json").read_text())
    assert metadata["entrypoints"]["main"] == ["proof.ExistingInitializer"]
