from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from minecraft_mod_ai import jdtls_bootstrap


def test_ensure_jdtls_reuses_executable_cache(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "managed-jdtls"
    launcher = root / "bin" / "jdtls"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("MMM_JDTLS_HOME", str(root))
    monkeypatch.delenv("MMM_JAVA_VERSION", raising=False)
    monkeypatch.setattr(jdtls_bootstrap, "_ensure_java_21", lambda: None)
    monkeypatch.setattr(
        jdtls_bootstrap,
        "_install_jdtls",
        lambda _root: pytest.fail("cache hit must not reinstall JDT LS"),
    )

    assert jdtls_bootstrap.ensure_jdtls() == launcher.resolve()


def test_expected_sha256_accepts_vendor_checksum_format(tmp_path: Path) -> None:
    digest = "ab" * 32
    checksum = tmp_path / "server.sha256"
    checksum.write_text(f"{digest}  archive.tar.gz\n", encoding="utf-8")

    assert jdtls_bootstrap._expected_sha256(checksum) == digest


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(jdtls_bootstrap.JDTLSBootstrapError, match="escapes"):
        jdtls_bootstrap._safe_extract(archive_path, tmp_path / "extract")


def test_safe_extract_jdk_rejects_escaping_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad-jdk.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("jdk/bin/java")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../outside"
        archive.addfile(info)

    with pytest.raises(jdtls_bootstrap.JDTLSBootstrapError, match="link escapes"):
        jdtls_bootstrap._safe_extract_jdk(archive_path, tmp_path / "extract")


def test_ensure_java_21_prefers_path_runtime_and_repairs_java_home(monkeypatch, tmp_path: Path) -> None:
    fake_java = tmp_path / "jdk-21" / "bin" / "java"
    fake_java.parent.mkdir(parents=True)
    fake_java.write_text("", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "jdk-17"))
    monkeypatch.delenv("MMM_PROJECT_JAVA_HOME", raising=False)
    monkeypatch.setattr(jdtls_bootstrap.shutil, "which", lambda _name: str(fake_java))
    monkeypatch.setattr(jdtls_bootstrap, "_java_major", lambda path: 21 if path == str(fake_java) else 17)

    jdtls_bootstrap._ensure_java_21()

    assert os.environ["JAVA_HOME"] == str(fake_java.resolve().parent.parent)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("25", 25), ("JavaSE-25", 25), ("21", 21)],
)
def test_requested_project_java_major_accepts_major_forms(
    monkeypatch, setting: str, expected: int
) -> None:
    monkeypatch.setenv("MMM_JAVA_VERSION", setting)

    assert jdtls_bootstrap._requested_project_java_major() == expected


def test_requested_project_java_major_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("MMM_JAVA_VERSION", "latest")

    with pytest.raises(jdtls_bootstrap.JDTLSBootstrapError, match="must be a Java major"):
        jdtls_bootstrap._requested_project_java_major()


def test_ensure_project_jdk_reuses_exact_major(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "jdk-25"
    java = home / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.write_text("", encoding="utf-8")
    java.chmod(0o755)
    monkeypatch.setenv("MMM_JAVA_VERSION", "25")
    monkeypatch.setattr(jdtls_bootstrap, "_find_matching_jdk", lambda major: home if major == 25 else None)
    monkeypatch.setattr(
        jdtls_bootstrap,
        "_install_project_jdk",
        lambda _major: pytest.fail("matching JDK must be reused"),
    )
    monkeypatch.setattr(jdtls_bootstrap, "_java_major", lambda path: 25 if path == str(java) else None)

    assert jdtls_bootstrap._ensure_project_jdk() == home.resolve()
    assert os.environ["MMM_PROJECT_JAVA_HOME"] == str(home.resolve())


def test_ensure_project_jdk_installs_missing_major(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "jdk-25"
    java = home / "bin" / "java"
    java.parent.mkdir(parents=True)
    java.write_text("", encoding="utf-8")
    java.chmod(0o755)
    monkeypatch.setenv("MMM_JAVA_VERSION", "25")
    monkeypatch.setattr(jdtls_bootstrap, "_find_matching_jdk", lambda _major: None)
    monkeypatch.setattr(jdtls_bootstrap, "_install_project_jdk", lambda major: home if major == 25 else None)
    monkeypatch.setattr(jdtls_bootstrap, "_java_major", lambda path: 25 if path == str(java) else None)

    assert jdtls_bootstrap._ensure_project_jdk() == home.resolve()
    assert os.environ["MMM_PROJECT_JAVA_HOME"] == str(home.resolve())


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("x86_64", "x64"), ("AMD64", "x64"), ("aarch64", "aarch64"), ("arm64", "aarch64")],
)
def test_adoptium_architecture_aliases(monkeypatch, machine: str, expected: str) -> None:
    monkeypatch.setattr(jdtls_bootstrap.platform, "machine", lambda: machine)

    assert jdtls_bootstrap._adoptium_architecture() == expected
