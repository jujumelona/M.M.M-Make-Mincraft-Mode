"""Current, compiler-resolved source must replace only its own old build ABI."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest

from minecraft_mod_ai.integrity_validators import validate_side
from minecraft_mod_ai.java_lsp import _resolve_project_java_home
from minecraft_mod_ai.java_symbol_resolver import JavaSymbolResolverError


PROGRAM = "src/main/java/demo/AuthoredProgram.java"
ANNOTATIONS = {
    "net/fabricmc/api/EnvType.java": "package net.fabricmc.api; public enum EnvType {CLIENT, SERVER}",
    "net/fabricmc/api/Environment.java": """package net.fabricmc.api;
@java.lang.annotation.Retention(java.lang.annotation.RetentionPolicy.CLASS)
public @interface Environment { EnvType value(); }
""",
}
CURRENT = "package demo; public class AuthoredProgram { public static int writeState(){return 19;} }"
FEATURE = "package demo; public class Feature { int run(){return AuthoredProgram.writeState();} }"


def _compile(tmp_path, sources, *, output="build/classes/java/main"):
    directory = tmp_path / "old-source"
    paths = []
    for relative, code in sources.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        paths.append(str(path))
    classes = tmp_path / output
    javac = Path(_resolve_project_java_home(17, require_compiler=True)) / "bin/javac"
    subprocess.run([str(javac), "--release", "17", "-d", str(classes), *paths],
                   check=True, capture_output=True, text=True)
    return classes


def _record(root, name, code):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return {"path": name, "sha256": hashlib.sha256(code.encode()).hexdigest(),
            "authority": "materialized_prerequisite_source"}


def _validate(root, source, classes, *, records=()):
    return validate_side(source, leaf_id="minecraft/event/callback", side="COMMON",
                         classpath=[classes], java_version="17", project_root=root, source_context=records)


def test_new_companion_method_replaces_only_its_own_stale_compiled_abi(tmp_path):
    classes = _compile(tmp_path, {"demo/AuthoredProgram.java":
        "package demo; public class AuthoredProgram { public static void oldEntry(){} }"})
    record = _record(tmp_path, PROGRAM, CURRENT)
    assert _validate(tmp_path, FEATURE, classes, records=[record])["status"] == "PASS"


def test_current_primary_source_replaces_its_previous_class_members(tmp_path):
    classes = _compile(tmp_path, {"demo/Feature.java": "package demo; public class Feature {}"})
    source = "package demo; public class Feature { int amount; int run(){this.amount=7;return this.amount;} }"
    assert _validate(tmp_path, source, classes)["status"] == "PASS"


@pytest.mark.parametrize("old_client,new_client", [(True, False), (False, True)])
def test_source_side_annotations_replace_the_old_local_class_side(tmp_path, old_client, new_client):
    annotation = "@net.fabricmc.api.Environment(net.fabricmc.api.EnvType.CLIENT) "
    body = "public class AuthoredProgram { public static int writeState(){return 19;} }"
    old = "package demo; " + (annotation if old_client else "") + body
    current = "package demo; " + (annotation if new_client else "") + body
    classes = _compile(tmp_path, {**ANNOTATIONS, "demo/AuthoredProgram.java": old})
    record = _record(tmp_path, PROGRAM, current)
    if new_client:
        with pytest.raises(ValueError, match="SIDE_VIOLATION"):
            _validate(tmp_path, FEATURE, classes, records=[record])
    else:
        assert _validate(tmp_path, FEATURE, classes, records=[record])["status"] == "PASS"


def test_removed_source_member_cannot_be_recovered_from_stale_bytecode(tmp_path):
    classes = _compile(tmp_path, {"demo/AuthoredProgram.java":
        "package demo; public class AuthoredProgram { public static int removed(){return 8;} }"})
    record = _record(tmp_path, PROGRAM, CURRENT)
    source = "package demo; public class Feature {int run(){return AuthoredProgram.removed();}}"
    with pytest.raises(JavaSymbolResolverError, match="JAVA_ANALYSIS_FAILED"):
        _validate(tmp_path, source, classes, records=[record])


@pytest.mark.parametrize("external_kind", ["jar", "another_project", "jar_in_output"])
def test_project_source_cannot_shadow_an_external_dependency_owner(tmp_path, external_kind):
    classes = _compile(tmp_path, {"demo/AuthoredProgram.java": CURRENT},
                       output="third-party/build/classes/java/main")
    if external_kind in {"jar", "jar_in_output"}:
        jar = tmp_path / ("build/classes/third-party.jar" if external_kind == "jar_in_output" else "third-party.jar")
        jar.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(jar, "w") as archive:
            for p in classes.rglob("*.class"):
                archive.write(p, p.relative_to(classes).as_posix())
        classes = jar
    record = _record(tmp_path, PROGRAM, CURRENT)
    with pytest.raises(JavaSymbolResolverError, match="JAVA_SOURCE_BINARY_OWNER_CONFLICT"):
        _validate(tmp_path, FEATURE, classes, records=[record])


def test_symlinked_build_directory_does_not_claim_external_bytecode_as_project_owned(tmp_path):
    external = _compile(tmp_path, {"demo/AuthoredProgram.java": CURRENT}, output="third-party/classes")
    build = tmp_path / "build"
    build.mkdir()
    (build / "classes").symlink_to(external, target_is_directory=True)
    record = _record(tmp_path, PROGRAM, CURRENT)
    with pytest.raises(JavaSymbolResolverError, match="JAVA_SOURCE_BINARY_OWNER_CONFLICT"):
        _validate(tmp_path, FEATURE, build / "classes", records=[record])


def test_an_external_binary_member_keeps_its_real_descriptor_and_side(tmp_path):
    source = "package api; public class Library { public static int answer(){return 23;} }"
    classes = _compile(tmp_path, {"api/Library.java": source}, output="external")
    good = "public class Feature {int run(){return api.Library.answer();}}"
    assert _validate(tmp_path, good, classes)["status"] == "PASS"
    with pytest.raises(JavaSymbolResolverError, match="JAVA_ANALYSIS_FAILED"):
        _validate(tmp_path, good.replace("answer()", "missing()"), classes)


def test_classpath_cannot_implicitly_admit_unrecorded_java_source(tmp_path):
    sourcepath = tmp_path / "unrecorded"
    path = sourcepath / "api/Hidden.java"
    path.parent.mkdir(parents=True)
    path.write_text("package api; public class Hidden { public static int value(){return 4;} }")
    source = "public class Feature { int run(){return api.Hidden.value();} }"
    with pytest.raises(JavaSymbolResolverError, match="JAVA_ANALYSIS_FAILED|JAVA_DECLARATION_SOURCE_UNVERIFIED"):
        _validate(tmp_path, source, sourcepath)


def test_binary_mismatch_identifies_abi_and_reference_source():
    from types import SimpleNamespace
    from minecraft_mod_ai.java_symbol_authority import resolved_declaration_side

    row = {"owner": "api/Library", "name": "write", "descriptor": "(I)V", "kind": "METHOD",
           "is_static": True, "side": "COMMON", "line": 23,
           "reference_source": "/src/main/java/demo/Feature.java", "declaration_source": ""}
    with pytest.raises(ValueError) as exc:
        resolved_declaration_side(row, {"api/Library": SimpleNamespace(members=(), side="COMMON")},
                                  {"api/Library": []}, source_names=set(), project_root=None)
    assert "api/Library#write(I)V" in str(exc.value)
    assert "origin=binary" in str(exc.value)
    assert "/src/main/java/demo/Feature.java:23" in str(exc.value)
