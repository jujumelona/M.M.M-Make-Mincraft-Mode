"""Reproducible real Fabric 1.20.1 item-registration compile/GameTest probe.

Downloads the official Gradle 8.6 wrapper and dependencies. Evidence covers this
single probe, not all canonical leaves or the complete supported version matrix.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_LEAF = "minecraft/item/registry"
_DEBUG_TARGET = "src/main/java/dev/mmm/debugfixture/DebugToken.java"

_PROBE_SOURCE = """package probe;
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
"""

_PROBE_TEST_SOURCE = """package probe;
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
"""

_BUILD_GRADLE = """plugins { id 'fabric-loom' version '1.6.12' }
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
"""

_DEBUG_SOURCE = """package dev.mmm.debugfixture;
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


class _DebugTokenRouter:
    """Deterministic coder transport for the real DebugToken compile probe."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._workspace: Path | None = None

    def bind_agent_workspace(self, workspace_root, *, require_fresh_evidence=True):
        assert require_fresh_evidence is True
        self._workspace = Path(workspace_root).resolve()

    def generate_text(self, role, messages, **kwargs):
        del messages
        assert role == "coder"
        assert kwargs.get("tool_stage") == "generation"
        assert kwargs.get("enable_tools") is False
        assert kwargs.get("response_format") == "json"
        assert isinstance(kwargs.get("response_schema"), dict)
        assert "output_token_ceiling" not in kwargs
        assert self._workspace is not None
        target = self._workspace / _DEBUG_TARGET
        if not target.is_file():
            raise AssertionError(
                "DebugToken host scaffold must exist before coder generation"
            )
        scaffold = target.read_text(encoding="utf-8")
        if "MMM_AUTHORED_FEATURE_BODY" not in scaffold:
            raise AssertionError(
                "DebugToken host scaffold marker is missing before coder generation"
            )
        return json.dumps(
            {
                "content": self._source,
                "summary": "Created the exact host-owned DebugToken source.",
            },
            ensure_ascii=False,
        )


def _probe_spec() -> dict[str, object]:
    return {
        "leaf_id": _LEAF,
        "context_id": "integrity-probe-1.20.1",
        "requirement": "Register the exact integrity_probe:verified_item item",
        "target_path": "src/main/java/probe/Probe.java",
        "language": "java",
        "operation": "CREATE_FILE",
        "side": "COMMON",
        "bindings": {"item": "integrity_probe:verified_item"},
        "output_schema": {"type": "string", "const": _PROBE_SOURCE},
        "render_mold": _PROBE_SOURCE,
        "slots": [],
    }


def _project_files(spec: dict[str, object], generated: dict[str, object]) -> dict[str, str]:
    fabric_mod = {
        "schemaVersion": 1,
        "id": "integrity_probe",
        "version": "1.0.0",
        "environment": "*",
        "entrypoints": {
            "main": ["probe.Probe"],
            "fabric-gametest": ["probe.ProbeTests"],
        },
        "depends": {
            "fabricloader": ">=0.15.11",
            "minecraft": "1.20.1",
            "java": ">=17",
            "fabric-api": "*",
        },
    }
    return {
        str(spec["target_path"]): str(generated["item_registry_artifact"]),
        "src/main/java/probe/ProbeTests.java": _PROBE_TEST_SOURCE,
        "settings.gradle": (
            "pluginManagement { repositories { maven { url='https://maven.fabricmc.net/' }; "
            "gradlePluginPortal() } }\nrootProject.name='integrity-probe'\n"
        ),
        "gradle.properties": "org.gradle.jvmargs=-Xmx2G\norg.gradle.workers.max=2\n",
        "build.gradle": _BUILD_GRADLE,
        "src/main/resources/fabric.mod.json": json.dumps(fabric_mod),
    }


def _materialize_project(project: Path, authority):
    spec = _probe_spec()
    generated = authority.executors["python_generator:" + _LEAF](
        {"item_identity": "integrity_probe:verified_item", "item_registry_input": spec},
        leaf_id=_LEAF,
        router=object(),
        authority=authority,
    )
    for name, content in _project_files(spec, generated).items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return spec, generated


def _run_debug_token_generation(root: Path, project: Path):
    from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
    from minecraft_mod_ai.complete_spec import CompleteProposal
    from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
    from minecraft_mod_ai.platform_catalog import adapter_for_target

    debug_plan = write_debug_example_plan(
        root / "debug-proposal.json",
        minecraft_version="1.20.1",
        loader="fabric",
    )
    proposal = CompleteProposal.from_dict(
        json.loads(debug_plan.read_text(encoding="utf-8"))
    )
    proposal.validate()
    module = proposal.modules[0]
    adapter = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(_DebugTokenRouter(_DEBUG_SOURCE)).generate(
        project,
        module=module,
        minecraft_version=adapter.minecraft_version,
        loader=adapter.loader,
        mappings=adapter.yarn_mappings,
    )
    target = project / _DEBUG_TARGET
    _assert_debug_generation_receipt(project, target, result)
    return module, result, target


def _assert_debug_generation_receipt(
    project: Path,
    target: Path,
    result: dict[str, object],
) -> None:
    if not target.is_file():
        raise AssertionError(
            "DebugToken custom-module transaction did not create the owned target"
        )
    if result.get("status") != "SOURCE_GENERATED":
        raise AssertionError(f"unexpected DebugToken generation status: {result!r}")
    relative = target.relative_to(project).as_posix()
    if relative not in set(result.get("touched_paths") or ()):
        raise AssertionError(
            "DebugToken generation receipt did not own the created target"
        )


def _download_gradle_wrapper_component(project: Path, name: str) -> None:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    url = "https://raw.githubusercontent.com/gradle/gradle/v8.6.0/" + name
    with urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def _ensure_gradle_wrapper(project: Path) -> None:
    names = ("gradlew", "gradlew.bat", "gradle/wrapper/gradle-wrapper.jar")
    with ThreadPoolExecutor(max_workers=len(names)) as executor:
        tuple(executor.map(lambda name: _download_gradle_wrapper_component(project, name), names))
    with urlopen(
        "https://services.gradle.org/distributions/gradle-8.6-bin.zip.sha256",
        timeout=60,
    ) as response:
        checksum = response.read().decode().strip()
    (project / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.6-bin.zip\n"
        + "distributionSha256Sum="
        + checksum
        + "\n",
        encoding="utf-8",
    )


def _run_real_fabric_evidence(
    root: Path,
    project: Path,
    spec: dict[str, object],
    generated: dict[str, object],
):
    from minecraft_mod_ai.evidence_store import EvidenceStore
    from minecraft_mod_ai.integrity_evidence import (
        binding_expectations,
        contract_hash,
        run_gradle_evidence,
        verify_execution_evidence,
    )
    from minecraft_mod_ai.populate_version_artifact_rules import make_implementation

    target = {
        "minecraft_version": "1.20.1",
        "java_version": "17",
        "fabric_loader": "0.15.11",
        "fabric_api": "0.92.2+1.20.1",
        "fabric_loom": "1.6.12",
        "gradle": "8.6",
        "loader": "fabric",
    }
    expected = binding_expectations(
        _LEAF,
        make_implementation(_LEAF, "1.20.1"),
        target,
    )
    receipt = generated["item_registry_receipt"]
    candidate = {
        "contract_sha256": contract_hash(spec),
        "target_path": spec["target_path"],
        "output_sha256": receipt["content_sha256"],
        "materialized_sha256": receipt["content_sha256"],
    }
    store = EvidenceStore(root / "evidence")
    evidence_id = run_gradle_evidence(
        project,
        store=store,
        expected=expected,
        gametest_task="runGameTestServer",
        report_glob="build/gametest-report.xml",
        timeout=900,
        candidate=candidate,
    )
    evidence_record = verify_execution_evidence(
        store,
        evidence_id,
        expected=expected,
    )
    (root / "result.json").write_text(
        json.dumps({"evidence_id": evidence_id, "expected": expected}, indent=2),
        encoding="utf-8",
    )
    return evidence_id, evidence_record


def _required_debug_class_digest(
    evidence_record: dict[str, object],
    class_file: str,
) -> object:
    artifacts = evidence_record.get("artifacts")
    if not isinstance(artifacts, dict) or class_file not in artifacts:
        raise AssertionError(
            "Isolated Gradle evidence passed but DebugToken.class was not captured"
        )
    return artifacts[class_file]


def _write_debug_e2e_receipt(
    root: Path,
    project: Path,
    module,
    result: dict[str, object],
    target: Path,
    evidence_id: str,
    evidence_record: dict[str, object],
) -> None:
    class_file = "build/classes/java/main/dev/mmm/debugfixture/DebugToken.class"
    class_digest = _required_debug_class_digest(evidence_record, class_file)
    payload = {
        "schema_version": "mmm/debug-token-compile-e2e-v1",
        "status": "PASS",
        "target": target.relative_to(project).as_posix(),
        "class_file": class_file,
        "class_sha256": class_digest,
        "generation_status": result.get("status"),
        "operation_count": result.get("operation_count"),
        "touched_paths": result.get("touched_paths"),
        "required_gates": module.required_gates,
        "fabric_evidence_id": evidence_id,
    }
    (root / "debug-token-e2e.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_output_root() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity

    root = _parse_output_root()
    project = root / "project"
    spec, generated = _materialize_project(project, bootstrap_integrity())
    module, debug_result, debug_target = _run_debug_token_generation(root, project)
    _ensure_gradle_wrapper(project)
    evidence_id, evidence_record = _run_real_fabric_evidence(
        root,
        project,
        spec,
        generated,
    )
    _write_debug_e2e_receipt(
        root,
        project,
        module,
        debug_result,
        debug_target,
        evidence_id,
        evidence_record,
    )
    print("Evidence:", evidence_id, flush=True)
    print("DebugToken fresh create -> real Gradle compile: PASS", flush=True)
    print("Real Fabric item registration compile and GameTest: PASS", flush=True)


if __name__ == "__main__":
    main()
