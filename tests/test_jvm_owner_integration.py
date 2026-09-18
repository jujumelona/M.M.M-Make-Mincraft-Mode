"""Real production-owner integration; opt in with MMM_JVM_OWNER_LIVE=1."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai.generation_verifier_resilience import (
    host_jdt_idle_timeout_seconds,
    host_jdt_startup_timeout_seconds,
)
from minecraft_mod_ai.jvm_owner_bootstrap import owner_command
from minecraft_mod_ai.owner_rpc import OwnerRPC


def _owner_command(tmp_path: Path, name: str) -> list[str]:
    """Launch the exact production owner bootstrap used by JavaCoreService."""

    return owner_command(tmp_path / f"owner-{name}")


@pytest.mark.skipif(os.environ.get("MMM_JVM_OWNER_LIVE") != "1", reason="real JVM integration opt-in")
def test_incremental_dependent_diagnostics(tmp_path):
    process = OwnerRPC(_owner_command(tmp_path, "incremental"))
    opened = False

    def rpc(method, params):
        nonlocal opened
        timeout = (
            host_jdt_startup_timeout_seconds()
            if method == "open" and not opened
            else host_jdt_idle_timeout_seconds()
        )
        result = process.request(method, params, timeout=timeout)
        if method == "open":
            opened = True
        return result
    source = tmp_path / "project/src"
    source.mkdir(parents=True)
    a = source / "A.java"
    a.write_text("public class A { public static int method() { return 1; } }", encoding="utf-8")
    (source / "B.java").write_text("public class B { int n = A.method(); }", encoding="utf-8")
    java_home = os.environ.get("JAVA_HOME", "C:/Program Files/Java/jdk-17")
    model = {"project_root": str(source.parent), "source_sets": [{"id": ":main", "project_path": ":", "name": "main",
             "source_roots": [str(source)], "classpath": [], "output_dirs": [], "java_home": java_home,
             "source_compatibility": "17", "target_compatibility": "17", "release": 17,
             "compiler_args": [], "annotation_processor_path": []}]}
    try:
        first = rpc("open", {"model": model})
        assert not [d for d in first["diagnostics"] if d["severity"] == "error"], first
        a.write_text("public class A {}", encoding="utf-8")
        broken = rpc("build", {"changes": [{"path": str(a), "operation": "modify"}], "full": False})
        assert any(d["path"].endswith("B.java") and d["severity"] == "error" for d in broken["diagnostics"]), broken
        a.write_text("public class A { public static int method() { return 1; } }", encoding="utf-8")
        fixed = rpc("build", {"changes": [{"path": str(a), "operation": "modify"}], "full": False})
        assert not [d for d in fixed["diagnostics"] if d["severity"] == "error"], fixed
        assert first["session_id"] == broken["session_id"] == fixed["session_id"]
        assert first["generation"] < broken["generation"] < fixed["generation"]
        a.unlink()
        deleted = rpc('build', {'changes': [], 'full': False})
        assert any(d['path'].endswith('B.java') and d['severity'] == 'error' for d in deleted['diagnostics'])
        a.write_text('public class A { public static int method() { return 1; } }', encoding='utf-8')
        recreated = rpc('build', {'changes': [], 'full': False})
        assert not [d for d in recreated['diagnostics'] if d['severity'] == 'error']
        rpc("close", {})
    finally:
        process.close()


@pytest.mark.skipif(os.environ.get('MMM_JVM_OWNER_LIVE') != '1', reason='real JVM integration opt-in')
def test_tooling_model_and_core_project_dependencies(tmp_path):
    from minecraft_mod_ai.java_core import JavaCoreService
    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock
    from minecraft_mod_ai.project_model import ResolvedBuildModel

    project = tmp_path / 'project'
    project.mkdir()
    target = adapter_for_target("1.20.1", "fabric")
    _write_platform_lock(project, target)
    (project / 'settings.gradle').write_text("rootProject.name='owner-test'\ninclude 'producer', 'consumer'\n")
    (project / 'build.gradle').write_text("subprojects { apply plugin: 'java' }\nproject(':consumer') { dependencies { implementation project(':producer') } }\n")
    producer = project / 'producer/src/main/java/A.java'
    consumer = project / 'consumer/src/main/java/B.java'
    for path in (producer, consumer):
        path.parent.mkdir(parents=True)
    producer.write_text('public class A { public static int method() { return 1; } }')
    consumer.write_text('public class B { int n = A.method(); }')
    params = JavaCoreService()._owner_resolve_parameters(project)
    with OwnerRPC(_owner_command(tmp_path, "tooling-model")) as rpc:
        raw = rpc.request(
            'resolve',
            params,
            timeout=host_jdt_startup_timeout_seconds(),
        )
        model = ResolvedBuildModel.from_dict(raw)
        assert model.gradle_version == target.gradle
        assert {source.project_path for source in model.source_sets} == {':producer', ':consumer'}
        opened = rpc.request(
            'open',
            {'model': model.to_dict()},
            timeout=host_jdt_startup_timeout_seconds(),
        )
        assert not [d for d in opened['diagnostics'] if d['severity'] == 'error'], opened
        producer.write_text('public class A {}')
        broken = rpc.request(
            'build',
            {'changes': [], 'full': False},
            timeout=host_jdt_idle_timeout_seconds(),
        )
        assert any(d['path'].endswith('B.java') and d['severity'] == 'error' for d in broken['diagnostics']), broken


@pytest.mark.skipif(os.environ.get('MMM_JVM_OWNER_LIVE') != '1', reason='real JVM integration opt-in')
def test_generation_service_preserves_errors_after_unchanged_check(tmp_path):
    from types import SimpleNamespace

    from minecraft_mod_ai import agent_tool_runtime
    from minecraft_mod_ai.generation_verifier_resilience import run_generation_verifier
    from minecraft_mod_ai.source_patch import TransactionalSourcePatcher

    from minecraft_mod_ai.platform_catalog import adapter_for_target
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock

    _write_platform_lock(tmp_path, adapter_for_target("1.20.1", "fabric"))
    (tmp_path / 'build.gradle').write_text("plugins { id 'java' }\n")
    (tmp_path / 'settings.gradle').write_text("rootProject.name='generation-owner'\n")
    patcher = TransactionalSourcePatcher(tmp_path)
    patcher.apply([{'operation': 'create', 'path': 'src/main/java/A.java', 'content': 'public class A { Missing value; }'}])
    runtime = SimpleNamespace(workspace_root=str(tmp_path))
    try:
        first = run_generation_verifier(runtime, {}, runtime_module=agent_tool_runtime)
        second = run_generation_verifier(runtime, {}, runtime_module=agent_tool_runtime)
        assert first['error_count'] > 0
        assert second['error_count'] == first['error_count']
        assert second['session_id'] == first['session_id']
        assert second['skipped'] is False
        assert second['generation'] > first['generation']
    finally:
        service = getattr(runtime, '_mmm_generation_java_service', None)
        if service is not None:
            service.close()


@pytest.mark.skipif(os.environ.get('MMM_JVM_OWNER_LIVE') != '1', reason='real JVM integration opt-in')
def test_core_runs_resolved_annotation_processor(tmp_path):
    processor = tmp_path / 'Generator.java'
    processor.write_text('''
import javax.annotation.processing.*;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.TypeElement;
import java.util.Set;
@SupportedAnnotationTypes("*")
@SupportedSourceVersion(SourceVersion.RELEASE_17)
public class Generator extends AbstractProcessor {
  private boolean done;
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment round) {
    if (!done && !round.processingOver()) {
      done = true;
      try (var writer = processingEnv.getFiler().createSourceFile("Generated").openWriter()) {
        writer.write("public class Generated {}");
      } catch (Exception e) { throw new RuntimeException(e); }
    }
    return false;
  }
}
''', encoding='utf-8')
    classes = tmp_path / 'processor-classes'
    classes.mkdir()
    subprocess.run(['javac', '-d', str(classes), str(processor)], check=True, timeout=30, capture_output=True)
    services = classes / 'META-INF/services/javax.annotation.processing.Processor'
    services.parent.mkdir(parents=True)
    services.write_text('Generator\n')
    jar = tmp_path / 'processor.jar'
    subprocess.run(['jar', 'cf', str(jar), '-C', str(classes), '.'], check=True, timeout=30, capture_output=True)
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'Main.java').write_text('public class Main { Generated value; }')
    model = {'source_sets': [{'id': ':main', 'name': 'main', 'project_path': ':',
              'source_roots': [str(source)], 'classpath': [], 'output_dirs': [],
              'java_home': os.environ.get('JAVA_HOME', 'C:/Program Files/Java/jdk-17'),
              'source_compatibility': '17', 'target_compatibility': '17', 'release': 17,
              'compiler_args': [], 'annotation_processor_path': [str(jar)]}]}
    with OwnerRPC(_owner_command(tmp_path, "annotation-processor")) as rpc:
        result = rpc.request(
            'open',
            {'model': model},
            timeout=host_jdt_startup_timeout_seconds(),
        )
        assert not [d for d in result['diagnostics'] if d['severity'] == 'error'], result



@pytest.mark.skipif(os.environ.get("MMM_JVM_OWNER_LIVE") != "1", reason="real JVM integration opt-in")
def test_debug_fixture_fabric_project_resolves_through_production_java_core(tmp_path):
    """Exercise the same Fabric/Loom project-model path used by Debug Mode."""

    from minecraft_mod_ai import fabric_immutable_rebind_contract
    from minecraft_mod_ai import fabric_official_template_provider as fabric_provider
    from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
    from minecraft_mod_ai.complete_spec import CompleteProposal
    from minecraft_mod_ai.java_core import JavaCoreService
    from minecraft_mod_ai.platform_catalog import adapter_for_lock_values

    plan = write_debug_example_plan(
        tmp_path / "debug-proposal.json",
        minecraft_version="1.20.1",
        loader="fabric",
    )
    proposal = CompleteProposal.from_dict(json.loads(plan.read_text(encoding="utf-8")))
    project = tmp_path / "fabric-project"
    adapter = adapter_for_lock_values(proposal.base_proposal.spec.platform)

    # Exercise the same provider-owned scaffold path used by production execution.
    fabric_immutable_rebind_contract.install()
    receipt = fabric_provider.bootstrap_fabric_project(
        project_root=project,
        spec=proposal.base_proposal.spec,
        adapter=adapter,
        cache_root=tmp_path / "fabric-provider-cache",
    )
    assert receipt["verified_generated_toolchain"]["minecraft_version"] == adapter.minecraft_version
    assert receipt["verified_generated_toolchain"]["gradle"] == adapter.gradle
    assert receipt.get("approval_rebind") == "EXACT"

    lock = project / ".minecraft_ai" / "platform-lock.json"
    assert lock.is_file(), "generated Fabric project must carry the target platform lock"

    service = JavaCoreService()
    try:
        receipt = service.diagnostics(
            project,
            timeout_seconds=300,
            full_scan=True,
        )
    finally:
        service.close()

    assert receipt["complete"] is True
    assert receipt["verification_backend"] == "jdt_core"
    assert receipt["verification_scope"] == "full"
    assert receipt["error_count"] == 0, receipt["diagnostics"]
    assert receipt["session_id"]
    assert receipt["model_id"]
