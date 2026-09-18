"""The owner runtime and project compile toolchain are intentionally distinct."""
from pathlib import Path

from minecraft_mod_ai import java_lsp, platform_catalog
from minecraft_mod_ai.java_core import JavaCoreService


def _jdk(tmp_path, major):
    home = tmp_path / f'jdk-{major}'
    home.mkdir()
    (home / 'release').write_text(f'JAVA_VERSION="{major}.0.1"\n')
    return home


def test_explicit_project_java_overrides_launcher(tmp_path, monkeypatch):
    host = _jdk(tmp_path, 17)
    project = _jdk(tmp_path, 25)
    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    monkeypatch.setenv('JAVA_HOME', str(host))
    monkeypatch.setattr(java_lsp, '_candidate_java_homes', lambda _major: [host, project])
    params = JavaCoreService._resolve_parameters(tmp_path)
    assert Path(params['java_home']) == project


def test_unavailable_project_jdk_is_provisioned_instead_of_falling_back_to_launcher(tmp_path, monkeypatch):
    host = _jdk(tmp_path, 17)
    project = _jdk(tmp_path, 25)
    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    monkeypatch.setattr(java_lsp, '_candidate_java_homes', lambda _major: [host])
    monkeypatch.setattr(
        java_lsp,
        '_provision_project_java_home',
        lambda required, detail: project.resolve(),
    )
    params = JavaCoreService._resolve_parameters(tmp_path)
    assert Path(params['java_home']) == project.resolve()
    assert Path(params['java_home']) != host.resolve()


def test_unconfigured_generic_gradle_project_keeps_own_toolchain(tmp_path, monkeypatch):
    monkeypatch.delenv('MMM_JAVA_VERSION', raising=False)
    assert JavaCoreService._resolve_parameters(tmp_path) == {'project_root': str(tmp_path)}


def test_platform_lock_does_not_force_gradle_runtime_to_project_target_jdk(tmp_path, monkeypatch):
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock

    target = platform_catalog.adapter_for_target('1.20.1', 'fabric')
    _write_platform_lock(tmp_path, target)
    stale = _jdk(tmp_path, 17)
    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    monkeypatch.setattr(java_lsp, '_candidate_java_homes', lambda _major: [stale])

    params = JavaCoreService._resolve_parameters(tmp_path)

    assert 'java_home' not in params
    assert params['gradle_version'] == target.gradle
    assert params['gradle_sha256'] == target.gradle_sha256



def test_platform_lock_resolves_gradle_coordinates_without_materialization(tmp_path, monkeypatch):
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock
    from minecraft_mod_ai.runner import GradleRunner

    target = platform_catalog.adapter_for_target('1.20.1', 'fabric')
    _write_platform_lock(tmp_path, target)
    monkeypatch.setattr(
        GradleRunner,
        'ensure_gradle',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('pure target resolution must not materialize Gradle')
        ),
    )

    params = JavaCoreService._resolve_parameters(tmp_path)

    assert params['gradle_version'] == target.gradle
    assert params['gradle_sha256'] == target.gradle_sha256
    assert 'java_home' not in params


def test_owner_resolution_materializes_exact_pinned_gradle(tmp_path, monkeypatch):
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock
    from minecraft_mod_ai.runner import GradleRunner

    target = platform_catalog.adapter_for_target('1.20.1', 'fabric')
    _write_platform_lock(tmp_path, target)
    gradle_home = tmp_path / 'verified-gradle'
    executable = gradle_home / 'bin' / 'gradle'
    executable.parent.mkdir(parents=True)
    executable.write_text('', encoding='utf-8')
    calls = []

    def ensure(self, version, sha256):
        calls.append((version, sha256))
        return executable

    monkeypatch.setattr(GradleRunner, 'ensure_gradle', ensure)

    params = JavaCoreService()._owner_resolve_parameters(tmp_path)

    assert calls == [(target.gradle, target.gradle_sha256)]
    assert params['gradle_home'] == str(gradle_home.resolve())
    assert params['gradle_user_home']
    assert 'gradle_version' not in params
    assert 'gradle_sha256' not in params
    assert 'java_home' not in params
