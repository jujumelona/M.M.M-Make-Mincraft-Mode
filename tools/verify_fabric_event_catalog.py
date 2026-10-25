"""Compile host-generated callback registrations against real Minecraft/Fabric.

This compiler probe uses synthetic neutral bodies, not model-generated gameplay.
It certifies callback discovery, host signatures and wiring, not game behavior.
All Minecraft classes and published libraries come from the official manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
from zipfile import ZipFile

from minecraft_mod_ai.construction_context import build_target_symbols
from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java
from minecraft_mod_ai.host_fabric_callback_abi import callback_interfaces


def _read(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _download(url, path, sha1):
    path = Path(path)
    if path.is_file() and hashlib.sha1(path.read_bytes()).hexdigest() == sha1:
        return path
    raw = _read(url)
    if hashlib.sha1(raw).hexdigest() != sha1:
        raise ValueError("Official artifact checksum mismatch: " + url)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".download")
    pending.write_bytes(raw)
    pending.replace(path)
    return path


def _maven_artifact(url, path):
    sha1 = _read(url + ".sha1").decode().strip().split()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", sha1):
        raise ValueError("Invalid Maven checksum: " + url)
    return _download(url, path, sha1)


def _compiler_probe_decision(data_name, parameters):
    """Answer only the neutral, compiler-checked predicate/call choices."""
    if "_predicate_try_" not in data_name and "_verified_call_try_" not in data_name:
        return None
    properties = parameters["properties"]
    if "verified_call_id" in properties:
        choices = properties["verified_call_id"]["enum"]
        assert choices, "The compiler probe requires a checked ABI candidate"
        return {"verified_call_id": choices[0]}
    # Older paths still support host-validated scalar predicates.
    if "_predicate_try_" in data_name and "expression" in properties:
        return {"expression": "true"}
    raise AssertionError("Compiler probe received an unknown predicate/call schema")


def _compiler_probe_operation(data_name, parameters):
    if "_required_state_abi_probe_count_try_" in data_name:
        return {"expression": "1"}
    if "_operation_0_try_" in data_name:
        choices = parameters["properties"]["action"]["enum"]
        assert "complete" in choices, "ABI probe requires its declared state writer"
        return {"action": "complete"}
    raise AssertionError("Unknown host-authored compiler probe operation: " + data_name)


class _CompilerProbeBodies:
    """Test-only ABI probe, NOT a gameplay-verified callback generator.

    Production now compiles callback conditions, effect operation frames, and
    return results separately. This neutral probe feeds those same authoring
    channels so javac proves their exact signatures against official jars.
    """

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        assert role == "coder"
        if data_name.startswith("select_callback_result_"):
            payload = json.loads(messages[1]["content"])
            callback = payload["verified_callback"]
            choices = parameters["properties"]["expression"].get("enum")
            if choices:
                result = choices[-1] if data_name.endswith("_handled") else choices[0]
            else:
                result_type = callback["return_type"]
                result = (
                    "true" if result_type == "boolean"
                    else "'a'" if result_type == "char"
                    else "0" if result_type in {
                        "byte", "short", "int", "long", "float", "double"
                    } else "null"
                )
            return {"expression": result}
        decision = _compiler_probe_decision(data_name, parameters)
        if decision is not None:
            return decision
        return _compiler_probe_operation(data_name, parameters)

    def generate_java_source(self, role, messages, *, data_name, source_frame):
        raise AssertionError("Compiler proof must not hand Java source ownership to a model")


def _callback_probe_source(event, symbols, name):
    spec = {"language": "java", "side": "COMMON",
            "target_path": "src/main/java/proof/" + name + ".java",
            "bindings": {"request_feature_entrypoint": "proof." + name}}
    state = {"owner": "proof.CompilerProbeState", "methods": [{
        "name": "writeState_abi_probe_count",
        "signature": "public static void writeState_abi_probe_count(int value);",
    }]}
    return compose_first_pass_java(
        _CompilerProbeBodies(), spec, [],
        construction_context={"symbols": symbols, "host_state_access": [state]},
        behaviors=[{"behavior_id": event["event_id"], "intent": "Compiler ABI probe",
                    "trigger": "The selected callback fires", "effects": "Compiler probe only",
                    "event_binding": event["event_id"], "state_writes": ["abi_probe_count"]}])


def _callback_probe_sources(root, events, symbols):
    directory = root / "sources" / "proof"
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "CompilerProbeState.java"
    state.write_text(
        "package proof; public final class CompilerProbeState {"
        "public static int count; "
        "public static void writeState_abi_probe_count(int value) {count=value;} }")
    paths = [str(state)]
    for index, event in enumerate(events):
        name = "Callback" + str(index)
        path = directory / (name + ".java")
        path.write_text(_callback_probe_source(event, symbols, name))
        paths.append(str(path))
    return paths


def verify(root, minecraft="26.2", fabric="0.155.2+26.2"):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError("JDK 25 javac is required; this proof cannot be skipped")
    version = subprocess.run([javac, "-version"], capture_output=True, text=True, check=True)
    if not re.search(r"javac (?:2[5-9]|[3-9]\d)(?:\.|\s|$)", version.stdout + version.stderr):
        raise RuntimeError("JDK 25 or later is required for actual Minecraft 26.2 classes")
    api_url = ("https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/"
               + fabric + "/fabric-api-" + fabric + ".jar")
    api_jar = _maven_artifact(api_url, root / ("fabric-api-" + fabric + ".jar"))
    modules = []
    with ZipFile(api_jar) as archive:
        mod = json.loads(archive.read("fabric.mod.json"))
        for name in archive.namelist():
            if name.startswith("META-INF/jars/") and name.endswith(".jar"):
                path = root / Path(name).name
                path.write_bytes(archive.read(name))
                modules.append(path)
    loader = re.search(r"\d+\.\d+\.\d+", mod["depends"]["fabricloader"])[0]
    loader_url = ("https://maven.fabricmc.net/net/fabricmc/fabric-loader/" + loader
                  + "/fabric-loader-" + loader + ".jar")
    loader_jar = _maven_artifact(loader_url, root / ("fabric-loader-" + loader + ".jar"))
    metadata_path = root / ("minecraft-" + minecraft + "-version.json")
    if not metadata_path.is_file():
        manifest = json.loads(_read("https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"))
        release = next(row for row in manifest["versions"] if row["id"] == minecraft)
        metadata_path.write_bytes(_read(release["url"]))
    metadata = json.loads(metadata_path.read_text())
    if metadata["id"] != minecraft:
        raise ValueError("Minecraft metadata identity mismatch")
    client = metadata["downloads"]["client"]
    minecraft_jar = _download(client["url"], root / ("minecraft-" + minecraft + "-client.jar"), client["sha1"])
    libraries = {row["downloads"]["artifact"]["path"]: row["downloads"]["artifact"]
                 for row in metadata["libraries"]
                 if "artifact" in row.get("downloads", {}) and "natives-" not in row["name"]}
    def download_library(row):
        return _download(row["url"], root / "libraries" / row["path"], row["sha1"])
    with ThreadPoolExecutor(max_workers=6) as pool:
        library_paths = list(pool.map(download_library, libraries.values()))
    # Include the exact Minecraft Java ABI as well as Fabric APIs. The same
    # proof checks the host's verified entity/attribute registry generator.
    symbols = build_target_symbols([minecraft_jar, *modules], java_version=25,
                                   namespace="OFFICIAL", classpath_sides={})
    events = callback_interfaces(symbols, side="COMMON")
    required = {
        "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents#END_SERVER_TICK",
        "net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents#AFTER",
        "net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents#AFTER_DAMAGE",
        "net.fabricmc.fabric.api.permission.v1.PermissionEvents#ON_REQUEST",
    }
    missing = required - {row["event_id"] for row in events}
    if missing:
        raise AssertionError("Regression: actual event fields missing: " + ", ".join(sorted(missing)))
    paths = _callback_probe_sources(root, events, symbols)
    # Compile the ACTUAL host-emitted living-entity registry against Minecraft
    # and Fabric classes. A synthetic signature stub could mask a 26.2 API
    # change, so both the class and registration are checked by javac.
    from minecraft_mod_ai.host_verified_entity_registry import verified_entity_registry_source
    probe_name = "CompiledCreature"
    entity_code = (
        "package proof;\n"
        "public class " + probe_name
        + " extends net.minecraft.world.entity.monster.Monster {\n"
        " public " + probe_name
        + "(net.minecraft.world.entity.EntityType<? extends net.minecraft.world.entity.monster.Monster> type,"
        + " net.minecraft.world.level.Level level) { super(type, level); }\n"
        + "}\n"
    )
    registry_spec = {
        "language": "java",
        "side": "COMMON",
        "bindings": {"module_config_json": json.dumps({
            "spawn_group": "creature", "entity_width": 0.9,
            "entity_height": 1.5,
        })},
    }
    registry_contract = {
        "canonical_leaf": "minecraft/entity/registry",
        "bindings": {"mod_id": "mmmprobe", "registry_path": "compiled_creature"},
    }
    registry_api = {"source": [{
        "authority": "inspected_materialized_source",
        "signature": "public class CompiledCreature extends net.minecraft.world.entity.monster.Monster",
        "source_sha256": hashlib.sha256(entity_code.encode()).hexdigest(),
    }]}
    registry_code = verified_entity_registry_source(
        registry_spec, registry_contract, registry_api, symbols,
        package="proof", name=probe_name + "Registry",
    )
    if registry_code is None:
        raise AssertionError("Host registry probe did not produce a Java compilation unit")
    if "FabricDefaultAttributeRegistry.register(ENTITY_TYPE" not in registry_code:
        raise AssertionError("Host registry omitted the required default Mob attributes")
    for name, source in (
        (probe_name, entity_code),
        (probe_name + "Registry", registry_code),
    ):
        filepath = root / "sources" / "proof" / (name + ".java")
        filepath.write_text(source, encoding="utf-8")
        paths.append(str(filepath))
    classpath = os.pathsep.join(map(str, [minecraft_jar, loader_jar, *modules, *library_paths]))
    compiled = subprocess.run([javac, "-proc:none", "-cp", classpath, "-d", str(root / "classes"), *paths],
                              capture_output=True, text=True, timeout=120)
    (root / "all-callbacks-javac.log").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError("Actual Fabric callback compilation failed:\n" + compiled.stderr)
    receipt = {"minecraft": minecraft, "fabric_api": fabric, "fabric_loader": loader,
               "common_event_count": len(events), "compiled_source_count": len(paths),
               "entity_registry_attributes_compiled": True,
               "javac_exit": compiled.returncode, "minecraft_sha1": client["sha1"],
               "gameplay_validated": False, "model_generation_validated": False}
    (root / "proof-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.cache_dir), indent=2))
