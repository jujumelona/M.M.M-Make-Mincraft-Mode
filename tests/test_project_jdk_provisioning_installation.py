from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import java_lsp
from minecraft_mod_ai import project_jdk_provisioning_installation as installation


class _BootstrapError(RuntimeError):
    pass


def _fake_java_lsp(home: Path, *, required: int = 25) -> SimpleNamespace:
    def resolve(requested: int | None = None) -> Path:
        major = required if requested is None else requested
        configured = os.environ.get("MMM_PROJECT_JAVA_HOME", "").strip()
        if not configured:
            raise _BootstrapError(f"no project JDK matching MMM_JAVA_VERSION={major}")
        return Path(configured).resolve()

    return SimpleNamespace(
        _resolve_project_java_home=resolve,
        _requested_project_java_major=lambda: required,
        JDTWorkspaceBootstrapError=_BootstrapError,
    )


def test_package_installs_late_project_jdk_resolver() -> None:
    assert getattr(
        java_lsp._resolve_project_java_home,
        installation._MARKER,
        False,
    ) is True


def test_late_resolver_provisions_major_known_after_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "jdk-25"
    java = home / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.write_text("", encoding="utf-8")
    java.chmod(0o755)
    monkeypatch.delenv("MMM_PROJECT_JAVA_HOME", raising=False)
    monkeypatch.setattr(
        installation.jdtls_bootstrap,
        "_install_project_jdk",
        lambda major: home if major == 25 else pytest.fail(f"unexpected major {major}"),
    )
    monkeypatch.setattr(
        installation.jdtls_bootstrap,
        "_java_major",
        lambda path: 25 if path == str(java) else None,
    )

    module = _fake_java_lsp(home)
    installation.install(module)

    assert module._resolve_project_java_home() == home.resolve()
    assert os.environ["MMM_PROJECT_JAVA_HOME"] == str(home.resolve())


def test_late_resolver_does_not_provision_when_exact_jdk_is_visible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "jdk-25"
    monkeypatch.setenv("MMM_PROJECT_JAVA_HOME", str(home))
    monkeypatch.setattr(
        installation,
        "_provision_exact_project_jdk",
        lambda _major: pytest.fail("visible exact JDK must be reused"),
    )

    module = _fake_java_lsp(home)
    installation.install(module)

    assert module._resolve_project_java_home(25) == home.resolve()


def test_late_resolver_rejects_wrong_major_after_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "jdk-21"
    java = home / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.write_text("", encoding="utf-8")
    java.chmod(0o755)
    monkeypatch.delenv("MMM_PROJECT_JAVA_HOME", raising=False)
    monkeypatch.setattr(installation.jdtls_bootstrap, "_install_project_jdk", lambda _major: home)
    monkeypatch.setattr(
        installation.jdtls_bootstrap,
        "_java_major",
        lambda path: 21 if path == str(java) else None,
    )

    module = _fake_java_lsp(home)
    installation.install(module)

    with pytest.raises(_BootstrapError, match="exact project JDK provisioning failed"):
        module._resolve_project_java_home(25)

    assert "MMM_PROJECT_JAVA_HOME" not in os.environ
