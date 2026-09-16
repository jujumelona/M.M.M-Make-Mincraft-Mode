"""The owner must bind Gradle to the selected project JDK, not its launcher."""
from pathlib import Path

import pytest

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


def test_unavailable_project_jdk_cannot_fall_back_to_launcher(tmp_path, monkeypatch):
    host = _jdk(tmp_path, 17)
    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    monkeypatch.setattr(java_lsp, '_candidate_java_homes', lambda _major: [host])
    with pytest.raises(java_lsp.JDTWorkspaceBootstrapError, match='25'):
        JavaCoreService._resolve_parameters(tmp_path)


def test_unconfigured_generic_gradle_project_keeps_own_toolchain(tmp_path, monkeypatch):
    monkeypatch.delenv('MMM_JAVA_VERSION', raising=False)
    assert JavaCoreService._resolve_parameters(tmp_path) == {'project_root': str(tmp_path)}


def test_platform_lock_wins_over_stale_environment(tmp_path, monkeypatch):
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock

    target = platform_catalog.adapter_for_target('1.20.1', 'fabric')
    _write_platform_lock(tmp_path, target)
    home = _jdk(tmp_path, 17)
    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    monkeypatch.setattr(java_lsp, '_candidate_java_homes', lambda _major: [home])
    assert Path(JavaCoreService._resolve_parameters(tmp_path)['java_home']) == home
