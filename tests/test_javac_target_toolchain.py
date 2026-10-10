"""The Compiler Tree bridge must run on the target JDK, independent of PATH."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai import java_lsp, javac_bridge, jdtls_bootstrap


def _jdk(root, major):
    home = root / f"jdk-{major}"
    (home / "bin").mkdir(parents=True)
    suffix = ".exe" if os.name == "nt" else ""
    for name in ("java", "javac"):
        (home / "bin" / (name + suffix)).write_text("")
    (home / "release").write_text(f'JAVA_VERSION="{major}.0.1"\n')
    return home


@pytest.mark.parametrize("parse_only", [True, False])
@pytest.mark.parametrize("provision", [True, False])
def test_bridge_uses_exact_target_jdk_for_parse_and_symbols(monkeypatch, tmp_path, parse_only, provision):
    old = _jdk(tmp_path, 17)
    target = _jdk(tmp_path, 25)
    suffix = ".exe" if os.name == "nt" else ""
    monkeypatch.setattr(shutil, "which", lambda name: str(old / "bin" / (name + suffix)))
    monkeypatch.setattr(java_lsp, "_candidate_java_homes", lambda major: [old] if provision else [old, target])
    provisioned = []

    def ensure(major):
        provisioned.append(major)
        return target

    monkeypatch.setattr(jdtls_bootstrap, "ensure_project_jdk", ensure)
    launched = []

    def run(command, **kwargs):
        launched.append(command)
        if command[0] != str(target / "bin" / ("java" + suffix)):
            return subprocess.CompletedProcess(command, 1, "", "release version 25 not supported")
        # Compiler Tree owns a dedicated result.json path. stdout can contain
        # unrelated JVM startup noise and must never be used as the API result.
        Path(command[-1]).write_text("[]", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "JVM startup warning", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert javac_bridge.analyze_java("class T {}", java_version="25", parse_only=parse_only) == []
    assert provisioned == ([25] if provision else [])
    assert launched[0][2:4] == ["parse" if parse_only else "resolve", "25"]


def test_bridge_still_rejects_invalid_java_with_real_target_jdk():
    if not shutil.which("javac"):
        pytest.skip("JDK required")
    with pytest.raises(ValueError, match="JAVA_ANALYSIS_FAILED"):
        javac_bridge.analyze_java("class T { void f( }", java_version="17", parse_only=True)


def test_java25_analysis_with_older_java_on_path():
    from pathlib import Path
    target = Path.home() / ".jdks" / "mmm-temurin-25"
    if not (target / "bin" / ("javac.exe" if os.name == "nt" else "javac")).is_file():
        pytest.skip("Java 25 JDK not installed")
    assert javac_bridge.analyze_java("class T {}", java_version="25", parse_only=True) == []
    rows = javac_bridge.analyze_java(
        "class T { String f() { return String.valueOf(1); } }", java_version="25",
    )
    assert any(row["owner"] == "java/lang/String" and row["name"] == "valueOf" for row in rows)
    with pytest.raises(ValueError, match="JAVA_ANALYSIS_FAILED"):
        javac_bridge.analyze_java("class T { void f( }", java_version="25", parse_only=True)


def test_same_version_jre_is_replaced_by_full_jdk(monkeypatch, tmp_path):
    jre = _jdk(tmp_path, 25)
    (jre / "bin" / ("javac.exe" if os.name == "nt" else "javac")).unlink()
    full = _jdk(tmp_path / "provisioned", 25)
    monkeypatch.setattr(java_lsp, "_candidate_java_homes", lambda major: [jre])
    monkeypatch.setattr(jdtls_bootstrap, "_candidate_jdk_homes", lambda: [jre])
    monkeypatch.setattr(jdtls_bootstrap, "_java_major", lambda executable: 25)
    monkeypatch.setattr(os, "access", lambda *args: True)
    assert jdtls_bootstrap._find_matching_jdk(25) is None
    calls = []

    def ensure(major):
        calls.append(major)
        return full

    monkeypatch.setattr(jdtls_bootstrap, "ensure_project_jdk", ensure)
    assert java_lsp._resolve_project_java_home(25, require_compiler=True) == full.resolve()
    assert calls == [25]
