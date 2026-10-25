"""Public resolver entry points must carry explicit project ownership."""
from pathlib import Path
import subprocess

import pytest

from minecraft_mod_ai.java_lsp import _resolve_project_java_home
from minecraft_mod_ai.java_symbol_resolver import (
    resolve_java_symbols_with_javac, resolve_java_symbols_with_javaparser,
)


@pytest.mark.parametrize("backend", ["javac", "javaparser"])
def test_public_resolver_can_replace_its_projects_stale_output(tmp_path, backend):
    old = tmp_path / "old-source/Feature.java"
    old.parent.mkdir()
    old.write_text("public class Feature {}")
    classes = tmp_path / "build/classes/java/main"
    javac = Path(_resolve_project_java_home(17, require_compiler=True)) / "bin/javac"
    subprocess.run([str(javac), "--release", "17", "-d", str(classes), str(old)],
                   check=True, capture_output=True)
    source = "public class Feature {int amount; int run(){amount=7;return amount;}}"
    current = tmp_path / "src/main/java/Feature.java"
    current.parent.mkdir(parents=True)
    current.write_text(source)
    if backend == "javac":
        symbols = resolve_java_symbols_with_javac(current, [classes], tmp_path / "unused-output",
                                                  project_root=tmp_path)
    else:
        symbols = resolve_java_symbols_with_javaparser(source, [classes], project_root=tmp_path)
    assert any(row.owner == "Feature" and row.name == "amount" and row.descriptor == "I" for row in symbols)
