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


class _DebugTokenRouter:
    """Deterministic coder transport for the real DebugToken compile probe."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._workspace: Path | None = None

    def bind_agent_workspace(self, workspace_root, *, require_fresh_evidence=True):
        assert require_fresh_evidence is True
        self._workspace = Path(workspace_root).resolve()

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs.get("tool_stage") == "generation"
        assert kwargs.get("enable_tools") is True
        assert kwargs.get("response_format") == "text"
        assert self._workspace is not None
        target = self._workspace / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise AssertionError("DebugToken fresh target unexpectedly existed before coder action")
        target.write_text(self._source, encoding="utf-8")
        return json.dumps(
            {"summary": "Created the exact host-owned DebugToken source."},
            ensure_ascii=False,
        )


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

    # Exercise the exact deterministic Debug Mode task through the same custom-module
    # staging/transaction path used by production, then let the real Gradle evidence
    # below prove that the created Java source is part of the compiled target.
    from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
    from minecraft_mod_ai.complete_spec import CompleteProposal
    from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator

    debug_plan = write_debug_example_plan(
        root / "debug-proposal.json",
        minecraft_version="1.20.1",
        loader="fabric",
    )
    debug_proposal = CompleteProposal.from_dict(
        json.loads(debug_plan.read_text(encoding="utf-8"))
    )
    debug_proposal.validate()
    debug_module = debug_proposal.modules[0]
    debug_source = """package dev.mmm.debugfixture;
import net.minecraft.item.Item;
import net.minecraft.registry.Registries;
import net.minecraft.registry.Registry;
import net.minecraft.util.Identifier;

public final class DebugToken {
    public static final Item ITEM = Registry.register(
        Registries.ITEM,
        new Identifier("integrity_probe", "debug_token"),
        new Item(new Item.Settings())
    );

    private DebugToken() {}
}
"""
    debug_router = _DebugTokenRouter(debug_source)
    debug_result = CustomModuleGenerator(debug_router).generate(
        project,
        module=debug_module,
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="1.20.1+build.10",
    )
    debug_target = project / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    if not debug_target.is_file():
        raise AssertionError("DebugToken custom-module transaction did not create the owned target")
    if debug_result.get("status") != "SOURCE_GENERATED":
        raise AssertionError(f"unexpected DebugToken generation status: {debug_result!r}")
    if debug_target.relative_to(project).as_posix() not in set(debug_result.get("touched_paths") or ()):
        raise AssertionError("DebugToken generation receipt did not own the created target")

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
    debug_class = (
        project
        / "build/classes/java/main/dev/mmm/debugfixture/DebugToken.class"
    )
    if not debug_class.is_file():
        raise AssertionError(
            "Gradle build passed but DebugToken.class was not produced from the fresh target"
        )
    (root / "debug-token-e2e.json").write_text(
        json.dumps(
            {
                "schema_version": "mmm/debug-token-compile-e2e-v1",
                "status": "PASS",
                "target": debug_target.relative_to(project).as_posix(),
                "class_file": debug_class.relative_to(project).as_posix(),
                "generation_status": debug_result.get("status"),
                "operation_count": debug_result.get("operation_count"),
                "touched_paths": debug_result.get("touched_paths"),
                "required_gates": debug_module.required_gates,
                "fabric_evidence_id": evidence_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("DebugToken fresh create -> real Gradle compile: PASS", flush=True)
    print("Real Fabric item registration compile and GameTest: PASS", flush=True)


if __name__ == "__main__":
    main()
