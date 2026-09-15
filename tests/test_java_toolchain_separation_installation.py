from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.java_toolchain_separation_installation import install


def _fake_java_lsp():
    class BootstrapError(RuntimeError):
        pass

    def base_configuration(project_java_home: Path | None = None):
        home = str(Path(project_java_home).resolve()) if project_java_home is not None else "/project/jdk-21"
        return {
            "java": {
                "autobuild": {"enabled": True},
                "configuration": {
                    "runtimes": [
                        {"name": "JavaSE-21", "path": home, "default": True},
                    ]
                },
                "import": {"gradle": {"enabled": True}},
            }
        }

    return SimpleNamespace(
        _jdt_configuration=base_configuration,
        JDTWorkspaceBootstrapError=BootstrapError,
    )


def test_gradle_daemon_uses_project_jdk_not_host_java_home(monkeypatch, tmp_path):
    project_jdk = tmp_path / "project-jdk-25"
    host_jdk = tmp_path / "host-jdk-21"
    monkeypatch.setenv("JAVA_HOME", str(host_jdk))

    java_lsp = _fake_java_lsp()
    install(java_lsp)
    configuration = java_lsp._jdt_configuration(project_jdk)

    assert configuration["java"]["configuration"]["runtimes"][0]["path"] == str(project_jdk.resolve())
    assert configuration["java"]["import"]["gradle"]["java"]["home"] == str(project_jdk.resolve())
    assert configuration["java"]["import"]["gradle"]["java"]["home"] != str(host_jdk.resolve())


def test_install_is_idempotent(tmp_path):
    java_lsp = _fake_java_lsp()
    install(java_lsp)
    first = java_lsp._jdt_configuration
    install(java_lsp)

    assert java_lsp._jdt_configuration is first
    assert java_lsp._jdt_configuration(tmp_path / "jdk")["java"]["import"]["gradle"]["java"]["home"] == str((tmp_path / "jdk").resolve())


def test_existing_gradle_configuration_is_preserved(tmp_path):
    java_lsp = _fake_java_lsp()
    install(java_lsp)

    configuration = java_lsp._jdt_configuration(tmp_path / "jdk")

    assert configuration["java"]["import"]["gradle"]["enabled"] is True
    assert configuration["java"]["autobuild"]["enabled"] is True
