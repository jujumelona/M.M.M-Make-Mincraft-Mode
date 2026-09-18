from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import java_lsp
from minecraft_mod_ai.java_lsp_trace import TracedJavaLanguageService


def _fake_jdk(tmp_path: Path, name: str, version: str) -> Path:
    home = tmp_path / name
    (home / "bin").mkdir(parents=True)
    (home / "release").write_text(f'JAVA_VERSION="{version}"\n', encoding="utf-8")
    return home


def test_project_jdk_resolver_matches_mmm_java_version(monkeypatch, tmp_path: Path) -> None:
    jdk17 = _fake_jdk(tmp_path, "jdk-17", "17.0.12")
    jdk21 = _fake_jdk(tmp_path, "jdk-21", "21.0.4")
    monkeypatch.setenv("MMM_JAVA_VERSION", "21")
    monkeypatch.setattr(java_lsp, "_candidate_java_homes", lambda _major: [jdk17, jdk21])

    assert java_lsp._resolve_project_java_home() == jdk21.resolve()
    runtime = java_lsp._project_java_runtime(jdk21.resolve())
    assert runtime == {"name": "JavaSE-21", "path": str(jdk21.resolve()), "default": True}


def test_missing_matching_project_jdk_provisions_exact_version_before_jdt_start(monkeypatch, tmp_path: Path) -> None:
    jdk21 = _fake_jdk(tmp_path, "jdk-21", "21.0.4")
    jdk17 = _fake_jdk(tmp_path, "jdk-17", "17.0.12")
    monkeypatch.setenv("MMM_JAVA_VERSION", "17")
    monkeypatch.setattr(java_lsp, "_candidate_java_homes", lambda _major: [jdk21])

    calls: list[tuple[int, str]] = []

    def provision(required: int, detail: str) -> Path:
        calls.append((required, detail))
        return jdk17.resolve()

    monkeypatch.setattr(java_lsp, "_provision_project_java_home", provision)

    assert java_lsp._resolve_project_java_home() == jdk17.resolve()
    assert calls and calls[0][0] == 17
    assert "21" in calls[0][1]


def test_launcher_java_and_project_java_are_separate(monkeypatch, tmp_path: Path) -> None:
    project = _fake_jdk(tmp_path, "project-17", "17.0.12")
    launcher = _fake_jdk(tmp_path, "launcher-21", "21.0.4")
    executable = launcher / "bin" / ("java.exe" if java_lsp.os.name == "nt" else "java")
    executable.write_text("", encoding="utf-8")
    monkeypatch.setenv("MMM_JAVA_VERSION", "17")
    monkeypatch.setenv("MMM_JDTLS_JAVA_HOME", str(launcher))

    config = java_lsp._jdt_configuration(project.resolve())
    environment = java_lsp._jdtls_environment()

    runtime = config["java"]["configuration"]["runtimes"][0]
    assert runtime["path"] == str(project.resolve())
    assert runtime["name"] == "JavaSE-17"
    assert environment["JAVA_HOME"] == str(launcher.resolve())
    assert environment["JAVA_HOME"] != runtime["path"]


def test_service_ready_cannot_bypass_semantic_probe(monkeypatch, tmp_path: Path) -> None:
    project = _fake_jdk(tmp_path, "project-17", "17.0.12")
    monkeypatch.setenv("MMM_JAVA_VERSION", "17")
    monkeypatch.delenv("MMM_JDTLS_JAVA_HOME", raising=False)
    monkeypatch.setattr(java_lsp, "_resolve_project_java_home", lambda required_major=None: project.resolve())

    class FakeRpc:
        def __init__(self, *args, **kwargs):
            self.process = SimpleNamespace(poll=lambda: None)
            self.workspace_folders = [{"uri": tmp_path.resolve().as_uri(), "name": tmp_path.name}]
            self.stderr = []
            self.messages = queue.Queue()
            self.closed = False

        def request(self, method, params, timeout):
            assert method == "initialize"
            return {"capabilities": {}}

        def notify(self, method, params):
            return None

        def close(self):
            self.closed = True

    fake = FakeRpc()
    monkeypatch.setattr(java_lsp, "_JsonRpcProcess", lambda *args, **kwargs: fake)

    def fail_probe(*args, **kwargs):
        raise java_lsp.JDTWorkspaceBootstrapError("semantic probe failed")

    monkeypatch.setattr(java_lsp, "_await_java_core_ready", fail_probe)
    service = java_lsp.JavaLanguageService(command="jdtls")

    with pytest.raises(java_lsp.JDTWorkspaceBootstrapError, match="semantic probe failed"):
        service._ensure_rpc_locked(tmp_path.resolve(), timeout_seconds=5)
    assert service.ready is False
    assert fake.closed is True


def test_traced_service_cannot_override_canonical_readiness() -> None:
    assert "_ensure_rpc_locked" not in TracedJavaLanguageService.__dict__
    assert TracedJavaLanguageService._ensure_rpc_locked is java_lsp.JavaLanguageService._ensure_rpc_locked


def test_semantic_probe_uses_compiler_diagnostics_without_hover(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "main" / "java"
    source_root.mkdir(parents=True)

    class FakeRpc:
        def __init__(self):
            self.messages = queue.Queue()
            self.process = SimpleNamespace(poll=lambda: None)
            self.stderr = []
            self.requests = []
            self.notifications = []

        def notify(self, method, params):
            self.notifications.append(method)

        def request(self, method, params, timeout):
            self.requests.append((method, params))
            raise AssertionError("semantic readiness must not use request/hover")

    monkeypatch.setattr(java_lsp, "_collect_diagnostics", lambda *args, **kwargs: {
        (source_root / "__MmmJdtReadinessProbe.java").resolve(strict=False).as_uri(): []
    })
    rpc = FakeRpc()
    java_lsp._await_java_core_ready(
        rpc,
        tmp_path.resolve(),
        timeout_seconds=2,
        quiet_seconds=0,
    )
    assert rpc.requests == []
    assert rpc.notifications == ["textDocument/didOpen", "textDocument/didClose"]

def test_core_type_diagnostic_is_workspace_bootstrap_failure() -> None:
    diagnostics = {
        "file:///workspace/src/main/java/example/Broken.java": [
            {"severity": 1, "message": "The type java.lang.Object cannot be resolved. It is indirectly referenced from required .class files"},
            {"severity": 1, "message": "String cannot be resolved to a type"},
        ]
    }

    with pytest.raises(java_lsp.JDTWorkspaceBootstrapError, match="workspace bootstrap failure"):
        java_lsp._raise_on_java_core_bootstrap_failure(diagnostics)
