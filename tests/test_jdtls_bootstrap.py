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


def test_ensure_java_21_prefers_path_runtime_and_repairs_java_home(monkeypatch, tmp_path: Path) -> None:
    fake_java = tmp_path / "jdk-21" / "bin" / "java"
    fake_java.parent.mkdir(parents=True)
    fake_java.write_text("", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "jdk-17"))
    monkeypatch.setattr(jdtls_bootstrap.shutil, "which", lambda _name: str(fake_java))
    monkeypatch.setattr(jdtls_bootstrap, "_java_major", lambda path: 21 if path == str(fake_java) else 17)

    jdtls_bootstrap._ensure_java_21()

    assert os.environ["JAVA_HOME"] == str(fake_java.resolve().parent.parent)
