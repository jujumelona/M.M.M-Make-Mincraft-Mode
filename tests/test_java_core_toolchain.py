"""The owner runtime and project compile toolchain are intentionally distinct."""
import inspect
import threading

from pathlib import Path

from minecraft_mod_ai import java_lsp, platform_catalog
import minecraft_mod_ai.java_core as java_core_module
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



def test_java_core_forwards_verifier_budget_to_owner_bootstrap(tmp_path, monkeypatch):
    from minecraft_mod_ai import jvm_owner_bootstrap

    seen: list[float] = []

    def fake_owner_command(workspace, *, timeout_seconds=600):
        assert workspace.is_absolute()
        seen.append(float(timeout_seconds))
        return ["fake-owner"]

    class FakeRPC:
        def __init__(self, command):
            assert command == ["fake-owner"]

    monkeypatch.setattr(jvm_owner_bootstrap, "owner_command", fake_owner_command)
    monkeypatch.setattr(java_core_module, "OwnerRPC", FakeRPC)

    service = JavaCoreService()
    service._prepare_project(tmp_path.resolve(), 123.5)

    assert seen == [123.5]


def test_owner_gradle_materialization_honors_remaining_verifier_budget(
    tmp_path,
    monkeypatch,
):
    from minecraft_mod_ai.platform_generation_contract import _write_platform_lock
    from minecraft_mod_ai.runner import GradleRunner

    target = platform_catalog.adapter_for_target("1.20.1", "fabric")
    _write_platform_lock(tmp_path, target)
    gradle_home = tmp_path / "verified-gradle"
    executable = gradle_home / "bin" / "gradle"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    calls = []

    def ensure(self, version, sha256, *, lock_timeout_seconds=None):
        calls.append(
            (
                version,
                sha256,
                self.download_timeout_seconds,
                lock_timeout_seconds,
            )
        )
        return executable

    monkeypatch.setattr(GradleRunner, "ensure_gradle", ensure)

    params = JavaCoreService()._owner_resolve_parameters(
        tmp_path,
        timeout_seconds=47.9,
    )

    assert calls == [
        (target.gradle, target.gradle_sha256, 47, 47)
    ]
    assert params["gradle_home"] == str(gradle_home.resolve())


def test_jvm_owner_bootstrap_uses_one_remaining_deadline() -> None:
    from minecraft_mod_ai.jvm_owner_bootstrap import owner_command

    source = inspect.getsource(owner_command)

    assert "remaining_timeout()" in source
    assert "timeout_seconds=remaining_timeout()" in source
    assert "lock_timeout_seconds=gradle_budget" in source
    assert "timeout=remaining_timeout()" in source
    assert "timeout_seconds=600" not in source
    assert "timeout=600" not in source


def test_java_core_project_lock_wait_is_bounded(tmp_path) -> None:
    from minecraft_mod_ai.project_write_lock import project_write_lock

    outcome: list[str] = []
    ready = threading.Event()

    def contender() -> None:
        ready.set()
        try:
            with project_write_lock(tmp_path, timeout_seconds=0.05):
                outcome.append("acquired")
        except TimeoutError:
            outcome.append("timeout")

    with project_write_lock(tmp_path):
        thread = threading.Thread(target=contender)
        thread.start()
        assert ready.wait(timeout=1.0)
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert outcome == ["timeout"]
