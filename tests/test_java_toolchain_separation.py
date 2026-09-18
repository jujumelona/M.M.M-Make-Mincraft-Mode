from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import java_lsp


def test_gradle_daemon_uses_resolved_project_jdk(monkeypatch) -> None:
    project_home = "/opt/mmm/project-jdk-25"
    monkeypatch.setattr(
        java_lsp,
        "_project_java_runtime",
        lambda _home=None: {
            "name": "JavaSE-25",
            "path": project_home,
            "default": True,
        },
    )

    result = java_lsp._jdt_configuration()

    assert result["java"]["configuration"]["runtimes"][0]["path"] == project_home
    assert result["java"]["import"]["gradle"]["java"]["home"] == project_home


def test_explicit_project_jdk_is_forwarded_to_runtime_resolution(monkeypatch) -> None:
    explicit = Path("/opt/mmm/selected-jdk-25")
    observed = []

    def runtime(home=None):
        observed.append(home)
        return {"name": "JavaSE-25", "path": str(explicit), "default": True}

    monkeypatch.setattr(java_lsp, "_project_java_runtime", runtime)
    result = java_lsp._jdt_configuration(explicit)

    assert observed == [explicit]
    assert result["java"]["import"]["gradle"]["java"]["home"] == str(explicit)


def test_jdt_launcher_java_home_is_process_local_and_separate(monkeypatch, tmp_path) -> None:
    launcher = tmp_path / "jdt-launcher"
    binary = launcher / "bin" / ("java.exe" if __import__("os").name == "nt" else "java")
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")

    monkeypatch.setenv("MMM_JDTLS_JAVA_HOME", str(launcher))
    monkeypatch.setenv("JAVA_HOME", "/host/default-java")
    env = java_lsp._jdtls_environment()

    assert env["JAVA_HOME"] == str(launcher.resolve())
    assert str(launcher.resolve() / "bin") in env["PATH"].split(__import__("os").pathsep)[0]
