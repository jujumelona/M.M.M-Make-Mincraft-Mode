"""Real Equinox integration; opt in with MMM_JVM_OWNER_LIVE=1 after installDist."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai.owner_rpc import OwnerRPC


def _configuration(runtime, workspace):
    target = workspace / 'configuration'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runtime / 'configuration/config.ini', target / 'config.ini')
    return target


@pytest.mark.skipif(os.environ.get("MMM_JVM_OWNER_LIVE") != "1", reason="real JVM integration opt-in")
def test_incremental_dependent_diagnostics(tmp_path):
    runtime = Path(__file__).parents[1] / "minecraft_mod_ai/jvm_owner/build/install/owner"
    framework = next((runtime / "plugins").glob("org.eclipse.osgi-*.jar"))
    process = OwnerRPC(["java", "-cp", str(framework), "org.eclipse.core.runtime.adaptor.EclipseStarter", "-configuration", str(_configuration(runtime, tmp_path)),
                       "-data", str(tmp_path / "workspace"), "-application", "mmm.owner.application", "-nosplash"])
    def rpc(method, params):
        return process.request(method, params, timeout=45)
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
    from minecraft_mod_ai.project_model import ResolvedBuildModel

    runtime = Path(__file__).parents[1] / 'minecraft_mod_ai/jvm_owner/build/install/owner'
    framework = next((runtime / 'plugins').glob('org.eclipse.osgi-*.jar'))
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'settings.gradle').write_text("rootProject.name='owner-test'\ninclude 'producer', 'consumer'\n")
    (project / 'build.gradle').write_text("subprojects { apply plugin: 'java' }\nproject(':consumer') { dependencies { implementation project(':producer') } }\n")
    producer = project / 'producer/src/main/java/A.java'
    consumer = project / 'consumer/src/main/java/B.java'
    for path in (producer, consumer):
        path.parent.mkdir(parents=True)
    producer.write_text('public class A { public static int method() { return 1; } }')
    consumer.write_text('public class B { int n = A.method(); }')
    with OwnerRPC(['java', '-cp', str(framework), 'org.eclipse.core.runtime.adaptor.EclipseStarter',
                   '-configuration', str(_configuration(runtime, tmp_path)), '-data', str(tmp_path / 'workspace'),
                   '-application', 'mmm.owner.application', '-nosplash']) as rpc:
        raw = rpc.request('resolve', {'project_root': str(project)}, timeout=120)
        model = ResolvedBuildModel.from_dict(raw)
        assert {source.project_path for source in model.source_sets} == {':producer', ':consumer'}
        opened = rpc.request('open', {'model': model.to_dict()}, timeout=45)
        assert not [d for d in opened['diagnostics'] if d['severity'] == 'error'], opened
        producer.write_text('public class A {}')
        broken = rpc.request('build', {'changes': [], 'full': False}, timeout=45)
        assert any(d['path'].endswith('B.java') and d['severity'] == 'error' for d in broken['diagnostics']), broken


@pytest.mark.skipif(os.environ.get('MMM_JVM_OWNER_LIVE') != '1', reason='real JVM integration opt-in')
def test_generation_service_preserves_errors_after_unchanged_check(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from minecraft_mod_ai import agent_tool_runtime, jvm_owner_bootstrap
    from minecraft_mod_ai.generation_verifier_resilience import run_generation_verifier
    from minecraft_mod_ai.source_patch import TransactionalSourcePatcher

    runtime_dir = Path(__file__).parents[1] / 'minecraft_mod_ai/jvm_owner/build/install/owner'
    framework = next((runtime_dir / 'plugins').glob('org.eclipse.osgi-*.jar'))

    def command(workspace):
        return ['java', '-cp', str(framework), 'org.eclipse.core.runtime.adaptor.EclipseStarter',
                '-configuration', str(_configuration(runtime_dir, workspace)), '-data', str(workspace / 'data'),
                '-application', 'mmm.owner.application', '-nosplash']

    monkeypatch.setattr(jvm_owner_bootstrap, 'owner_command', command)
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
    runtime = Path(__file__).parents[1] / 'minecraft_mod_ai/jvm_owner/build/install/owner'
    framework = next((runtime / 'plugins').glob('org.eclipse.osgi-*.jar'))
    model = {'source_sets': [{'id': ':main', 'name': 'main', 'project_path': ':',
              'source_roots': [str(source)], 'classpath': [], 'output_dirs': [],
              'java_home': os.environ.get('JAVA_HOME', 'C:/Program Files/Java/jdk-17'),
              'source_compatibility': '17', 'target_compatibility': '17', 'release': 17,
              'compiler_args': [], 'annotation_processor_path': [str(jar)]}]}
    with OwnerRPC(['java', '-cp', str(framework), 'org.eclipse.core.runtime.adaptor.EclipseStarter',
                   '-configuration', str(_configuration(runtime, tmp_path)), '-data', str(tmp_path / 'workspace'),
                   '-application', 'mmm.owner.application', '-nosplash']) as rpc:
        result = rpc.request('open', {'model': model}, timeout=45)
        assert not [d for d in result['diagnostics'] if d['severity'] == 'error'], result
