"""The pinned Gradle bootstrap and the target-aware owner have separate JDKs."""
import os
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.jvm_owner_bootstrap import owner_command


def test_cold_owner_build_pins_compatible_jdk_without_changing_runtime_java(tmp_path, monkeypatch):
    build_home, runtime_home = tmp_path / 'jdk21', tmp_path / 'jdk25'
    for home in (build_home, runtime_home):
        (home / 'bin').mkdir(parents=True)
        (home / 'bin/java').write_text('fixture')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('JAVA_HOME', str(runtime_home))
    monkeypatch.setenv('PATH', str(runtime_home / 'bin'))
    resolutions, builds = [], []

    def resolve(major, **kwargs):
        resolutions.append(major)
        assert major == 21
        assert kwargs.get('require_compiler') is True
        return build_home

    def build(command, **kwargs):
        env = kwargs.get('env') or {}
        assert env.get('JAVA_HOME') == str(build_home)
        assert env['PATH'].split(os.pathsep)[0] == str(build_home / 'bin')
        assert '-Dorg.gradle.java.home=' + str(build_home) in command
        builds.append(command)
        target = kwargs['cwd'] / 'build/install/owner'
        (target / 'configuration').mkdir(parents=True)
        (target / 'configuration/config.ini').write_text('fixture')
        (target / 'plugins').mkdir()
        (target / 'plugins/org.eclipse.osgi-fixture.jar').write_text('fixture')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr('minecraft_mod_ai.java_lsp._resolve_project_java_home', resolve)
    monkeypatch.setattr('minecraft_mod_ai.runner.GradleRunner.ensure_gradle', lambda *a, **k: tmp_path / 'gradle')
    monkeypatch.setattr('minecraft_mod_ai.jvm_owner_bootstrap.subprocess.run', build)
    command = owner_command(tmp_path / 'workspace', java_home=runtime_home, required_major=25)
    assert command[0] == str(runtime_home / 'bin/java')
    assert os.environ['JAVA_HOME'] == str(runtime_home)
    again = owner_command(tmp_path / 'workspace-again', java_home=runtime_home, required_major=25)
    assert again[0] == command[0]
    assert resolutions == [21]
    assert len(builds) == 1
