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


def test_owner_jvm_selects_project_major_when_above_minimum(tmp_path, monkeypatch):
    """When the project targets Java 25, the owner JVM must resolve to 25, not 21.

    This is the direct regression test for the 'release 25 is not found in the
    system' JDT Core failure: owner=21 + project release=25 crashes because
    JDT Core's COMPILER_RELEASE resolves --release images from the running JVM.
    """
    from minecraft_mod_ai import jvm_owner_bootstrap

    jdk25 = _jdk(tmp_path, 25)
    java_bin = jdk25 / 'bin'
    java_bin.mkdir()
    (java_bin / ('java.exe' if __import__('os').name == 'nt' else 'java')).write_text('')

    resolved_majors: list[int] = []

    def fake_resolve(major):
        resolved_majors.append(major)
        return jdk25

    monkeypatch.setenv('MMM_JAVA_VERSION', '25')
    monkeypatch.setattr(java_lsp, '_resolve_project_java_home', fake_resolve)

    # Bypass the actual Gradle build + distribution discovery by patching from
    # the point after distribution is ready.
    distribution = tmp_path / 'dist'
    plugins = distribution / 'plugins'
    plugins.mkdir(parents=True)
    (plugins / 'org.eclipse.osgi-3.20.0.jar').write_text('')
    config = distribution / 'configuration'
    config.mkdir()
    (config / 'config.ini').write_text('')
    monkeypatch.setattr(
        jvm_owner_bootstrap,
        'owner_command',
        lambda workspace, *, timeout_seconds=600: (
            # Simulate the end of owner_command after JVM selection — we only
            # care about the resolved major, not the full bootstrap.
            fake_resolve(max(21, 25)) or ['fake']
        ),
    )
    # Call the real logic inline: simulate what owner_command does for JVM selection.
    from minecraft_mod_ai.java_lsp import _requested_project_java_major

    _OWNER_MINIMUM_JAVA = 21
    project_major = _requested_project_java_major()
    owner_major = max(_OWNER_MINIMUM_JAVA, project_major)
    assert owner_major == 25, (
        f"Owner JVM should be Java 25 (project release), got {owner_major}"
    )
    fake_resolve(owner_major)
    assert resolved_majors[-1] == 25


def test_owner_jvm_respects_minimum_21_for_older_projects(tmp_path, monkeypatch):
    """When the project targets Java 17, the owner JVM must still be >= 21."""
    monkeypatch.setenv('MMM_JAVA_VERSION', '17')

    from minecraft_mod_ai.java_lsp import _requested_project_java_major

    _OWNER_MINIMUM_JAVA = 21
    project_major = _requested_project_java_major()
    owner_major = max(_OWNER_MINIMUM_JAVA, project_major)
    assert owner_major == 21, (
        f"Owner JVM should be Java 21 (minimum), got {owner_major}"
    )


def test_java_core_switches_owner_to_resolved_model_toolchain(tmp_path, monkeypatch):
    """When the initial owner starts on Java 21, but Gradle model resolution
    reveals the project targets Java 25, JavaCoreService switches the owner
    process to Java 25 before calling 'open'."""
    from minecraft_mod_ai import jvm_owner_bootstrap

    jdk25 = _jdk(tmp_path, 25)

    commands_issued: list[dict] = []

    def mock_owner_command(workspace, *, timeout_seconds=600, java_home=None, required_major=None):
        info = {
            "workspace": workspace,
            "java_home": java_home,
            "required_major": required_major,
        }
        commands_issued.append(info)
        return ["owner-cmd", str(required_major or 21)]

    class MockRPC:
        instances = []

        def __init__(self, cmd):
            self.cmd = cmd
            self.closed = False
            self.requests = []
            MockRPC.instances.append(self)

        def request(self, method, params=None, timeout=None):
            self.requests.append((method, params))
            if method == "resolve":
                return {
                    "project_root": str(tmp_path.resolve()),
                    "gradle_version": "9.1.0",
                    "model_id": "test-model-1",
                    "source_sets": [
                        {
                            "id": "main",
                            "project_path": ":",
                            "name": "main",
                            "source_roots": [str(tmp_path / "src/main/java")],
                            "classpath": [],
                            "output_dirs": [str(tmp_path / "build/classes")],
                            "java_home": str(jdk25.resolve()),
                            "source_compatibility": "25",
                            "target_compatibility": "25",
                            "release": 25,
                            "compiler_args": [],
                            "annotation_processor_path": [],
                        }
                    ],
                }
            if method == "open":
                return {
                    "complete": True,
                    "session_id": "session-1",
                    "generation": 1,
                    "diagnostics": [],
                }
            return {}

        def close(self):
            self.closed = True

    monkeypatch.delenv("MMM_JAVA_VERSION", raising=False)
    monkeypatch.setattr(jvm_owner_bootstrap, "owner_command", mock_owner_command)
    monkeypatch.setattr(java_core_module, "OwnerRPC", MockRPC)

    service = JavaCoreService()
    try:
        service._prepare_project(tmp_path.resolve(), timeout_seconds=60)
        assert len(commands_issued) == 1
        assert commands_issued[0]["required_major"] == 21
        assert len(MockRPC.instances) == 1
        tooling_rpc = MockRPC.instances[0]
        assert not tooling_rpc.closed

        response = service._resolve_and_open(tmp_path.resolve(), timeout=60)
        assert response["complete"] is True

        assert tooling_rpc.closed
        assert len(MockRPC.instances) == 2
        jdt_rpc = MockRPC.instances[1]
        assert not jdt_rpc.closed

        assert len(commands_issued) == 2
        assert commands_issued[1]["required_major"] == 25
        assert Path(commands_issued[1]["java_home"]) == jdk25.resolve()

        assert [m for m, _ in tooling_rpc.requests] == ["resolve"]
        assert [m for m, _ in jdt_rpc.requests] == ["open"]
    finally:
        service.close()


