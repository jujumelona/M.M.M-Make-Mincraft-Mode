"""Reproducible real Fabric 1.20.1 item-registration compile/GameTest probe.

Downloads the official Gradle 8.6 wrapper and dependencies. Evidence covers this
single probe, not all canonical leaves or the complete supported version matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity
    from minecraft_mod_ai.populate_version_artifact_rules import make_implementation
    from minecraft_mod_ai.integrity_evidence import run_gradle_evidence, verify_execution_evidence, binding_expectations, contract_hash
    from minecraft_mod_ai.evidence_store import EvidenceStore
    authority = bootstrap_integrity()
    project = root / "project"
    leaf = "minecraft/item/registry"
    source = '''package probe;
import net.fabricmc.api.ModInitializer;
import net.minecraft.item.Item;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;
public class Probe implements ModInitializer {
    public static final Item ITEM = Registry.register(Registries.ITEM,
        new Identifier("integrity_probe", "verified_item"), new Item(new Item.Settings()));
    public void onInitialize() {}
}
'''
    spec = {"leaf_id": leaf, "context_id": "integrity-probe-1.20.1", "requirement": "Register the exact integrity_probe:verified_item item",
            "target_path": "src/main/java/probe/Probe.java", "language": "java", "operation": "CREATE_FILE",
            "side": "COMMON", "bindings": {"item": "integrity_probe:verified_item"},
            "output_schema": {"type": "string", "const": source}, "render_mold": source, "slots": []}
    generated = authority.executors["python_generator:"+leaf](
        {"item_identity": "integrity_probe:verified_item", "item_registry_input": spec},
        leaf_id=leaf, router=object(), authority=authority)
    files = {
        spec["target_path"]: generated["item_registry_artifact"],
        "src/main/java/probe/ProbeTests.java": '''package probe;
import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.minecraft.test.GameTest;
import net.minecraft.test.TestContext;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;
public class ProbeTests {
    @GameTest(templateName = FabricGameTest.EMPTY_STRUCTURE)
    public void itemRegistered(TestContext context) {
        if (Registries.ITEM.get(new Identifier("integrity_probe", "verified_item")) != Probe.ITEM)
            throw new AssertionError("Registered item identity mismatch");
        context.complete();
    }
}
''',
        "settings.gradle": "pluginManagement { repositories { maven { url='https://maven.fabricmc.net/' }; gradlePluginPortal() } }\nrootProject.name='integrity-probe'\n",
        "gradle.properties": "org.gradle.jvmargs=-Xmx2G\norg.gradle.workers.max=2\n",
        "build.gradle": '''plugins { id 'fabric-loom' version '1.6.12' }
version='1.0.0'
group='probe'
dependencies {
    minecraft 'com.mojang:minecraft:1.20.1'
    mappings 'net.fabricmc:yarn:1.20.1+build.10:v2'
    modImplementation 'net.fabricmc:fabric-loader:0.15.11'
    modImplementation 'net.fabricmc.fabric-api:fabric-api:0.92.2+1.20.1'
}
java { sourceCompatibility=JavaVersion.VERSION_17; targetCompatibility=JavaVersion.VERSION_17 }
loom { runs { gameTestServer {
    server()
    runDir 'build/gametest'
    vmArg '-Dfabric-api.gametest'
    vmArg "-Dfabric-api.gametest.report-file=${file('build/gametest-report.xml').absolutePath}"
} } }
''',
        "src/main/resources/fabric.mod.json": json.dumps({"schemaVersion": 1, "id": "integrity_probe", "version": "1.0.0",
            "environment": "*", "entrypoints": {"main": ["probe.Probe"], "fabric-gametest": ["probe.ProbeTests"]},
            "depends": {"fabricloader": ">=0.15.11", "minecraft": "1.20.1", "java": ">=17", "fabric-api": "*"}}),
    }
    for name, content in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    for name in ("gradlew", "gradlew.bat", "gradle/wrapper/gradle-wrapper.jar"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urlopen("https://raw.githubusercontent.com/gradle/gradle/v8.6.0/" + name, timeout=60) as response:
                path.write_bytes(response.read())
    with urlopen("https://services.gradle.org/distributions/gradle-8.6-bin.zip.sha256", timeout=60) as response:
        checksum = response.read().decode().strip()
    (project / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.6-bin.zip\n"
        + "distributionSha256Sum=" + checksum + "\n", encoding="utf-8")
    target = {"minecraft_version": "1.20.1", "java_version": "17", "fabric_loader": "0.15.11",
              "fabric_api": "0.92.2+1.20.1", "fabric_loom": "1.6.12", "gradle": "8.6", "loader": "fabric"}
    implementation = make_implementation(leaf, "1.20.1")
    expected = binding_expectations(leaf, implementation, target)
    store = EvidenceStore(root / "evidence")
    evidence_id = run_gradle_evidence(project, store=store, expected=expected,
        gametest_task="runGameTestServer", report_glob="build/gametest-report.xml", timeout=900,
        candidate={"contract_sha256": contract_hash(spec), "target_path": spec["target_path"],
                   "output_sha256": generated["item_registry_receipt"]["content_sha256"],
                   "materialized_sha256": generated["item_registry_receipt"]["content_sha256"]})
    (root / "result.json").write_text(json.dumps({"evidence_id": evidence_id, "expected": expected}, indent=2), encoding="utf-8")
    print("Evidence:", evidence_id, flush=True)
    verify_execution_evidence(store, evidence_id, expected=expected)
    print("Real Fabric item registration compile and GameTest: PASS", flush=True)


if __name__ == "__main__":
    main()
