"""Real Gradle/JDT regression for a project newer than the owner launcher JVM."""
import os
import shutil
from pathlib import Path

import pytest

from minecraft_mod_ai.java_core import JavaCoreService

pytestmark = pytest.mark.skipif(
    os.environ.get('MMM_JVM_OWNER_LIVE') != '1' or not os.environ.get('MMM_TEST_JDK25_HOME'),
    reason='requires installed owner runtime and MMM_TEST_JDK25_HOME',
)


def test_project_jdk_reaches_gradle_and_preserves_release_api_boundary(tmp_path, monkeypatch):
    from minecraft_mod_ai import jvm_owner_bootstrap

    runtime = Path(__file__).parents[1] / 'minecraft_mod_ai/jvm_owner/build/install/owner'
    framework = next((runtime / 'plugins').glob('org.eclipse.osgi-*.jar'))

    def command(workspace):
        config = workspace / 'configuration'
        config.mkdir()
        shutil.copy2(runtime / 'configuration/config.ini', config / 'config.ini')
        return ['java', '-cp', str(framework), 'org.eclipse.core.runtime.adaptor.EclipseStarter',
                '-configuration', str(config), '-data', str(workspace / 'data'),
                '-application', 'mmm.owner.application', '-nosplash']

    monkeypatch.setattr(jvm_owner_bootstrap, 'owner_command', command)
    monkeypatch.setenv('MMM_PROJECT_JAVA_HOME', os.environ['MMM_TEST_JDK25_HOME'])
    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    (tmp_path / 'settings.gradle').write_text("rootProject.name='release-boundary'\n")
    build = tmp_path / 'build.gradle'
    build.write_text("plugins { id 'java' }\ntasks.withType(JavaCompile).configureEach { options.release = 25 }\n")
    wrapper = tmp_path / 'gradle/wrapper/gradle-wrapper.properties'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('distributionUrl=https\\://services.gradle.org/distributions/gradle-9.1.0-bin.zip\n')
    source = tmp_path / 'src/main/java/Example.java'
    source.parent.mkdir(parents=True)
    source.write_text('public class Example { String first = java.util.List.of("ok").getFirst(); }')
    service = JavaCoreService()
    try:
        first = service.diagnostics(tmp_path, timeout_seconds=240)
        assert first['error_count'] == 0, first['diagnostics']
        assert Path(service._model.source_sets[0].java_home).samefile(os.environ['MMM_TEST_JDK25_HOME'])
        build.write_text("plugins { id 'java' }\ntasks.withType(JavaCompile).configureEach { options.release = 17 }\n")
        older = service.diagnostics(tmp_path, timeout_seconds=240)
        errors = [d for rows in older['diagnostics'].values() for d in rows if d['severity'] == 1]
        assert errors and all(d['path'].endswith('Example.java') for d in errors), older
        assert any('getFirst' in d['message'] for d in errors), errors
        source.write_text('public class Example { String first = java.util.List.of("ok").get(0); }')
        repaired = service.diagnostics(tmp_path, timeout_seconds=60)
        assert repaired['error_count'] == 0, repaired
        assert first['session_id'] == older['session_id'] == repaired['session_id']
    finally:
        service.close()
