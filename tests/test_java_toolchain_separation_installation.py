from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.java_lsp import _jdtls_environment
from minecraft_mod_ai.java_toolchain_separation_installation import install


class _BootstrapError(RuntimeError):
    pass


def _fake_java_lsp(runtime_home: str | None):
    def configuration(project_java_home=None):
        runtimes = []
        if runtime_home is not None:
            runtimes.append(
                {
                    "name": "JavaSE-25",
                    "path": runtime_home,
                    "default": True,
                }
            )
        return {
            "java": {
                "configuration": {"runtimes": runtimes},
                "import": {"gradle": {"enabled": True}},
            }
        }

    return SimpleNamespace(
        _jdt_configuration=configuration,
        JDTWorkspaceBootstrapError=_BootstrapError,
    )


def test_gradle_daemon_uses_project_jdk_from_jdt_runtime_not_host_java_home() -> None:
    project_home = "/opt/mmm/project-jdk-25"
    module = _fake_java_lsp(project_home)
    install(module)

    result = module._jdt_configuration()

    assert result["java"]["configuration"]["runtimes"][0]["path"] == project_home
    assert result["java"]["import"]["gradle"]["java"]["home"] == project_home


def test_explicit_project_jdk_is_used_for_gradle_daemon() -> None:
    module = _fake_java_lsp("/opt/mmm/default-project-jdk")
    install(module)
    explicit = Path("/opt/mmm/selected-jdk-25")

    result = module._jdt_configuration(explicit)

    assert result["java"]["import"]["gradle"]["java"]["home"] == str(explicit.resolve())


def test_missing_project_jdk_fails_closed_instead_of_inheriting_launcher_jvm() -> None:
    module = _fake_java_lsp(None)
    install(module)

    with pytest.raises(_BootstrapError, match="project JDK"):
        module._jdt_configuration()


def test_jdt_launcher_java_home_is_process_local_and_separate(
    monkeypatch,
    tmp_path,
) -> None:
    launcher = tmp_path / "jdt-launcher"
    binary = launcher / "bin" / ("java.exe" if __import__("os").name == "nt" else "java")
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")

    monkeypatch.setenv("MMM_JDTLS_JAVA_HOME", str(launcher))
    monkeypatch.setenv("JAVA_HOME", "/host/default-java")
    env = _jdtls_environment()

    assert env["JAVA_HOME"] == str(launcher.resolve())
    assert str(launcher.resolve() / "bin") in env["PATH"].split(__import__("os").pathsep)[0]
