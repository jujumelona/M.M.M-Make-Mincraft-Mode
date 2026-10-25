"""Regression: registry owner references a materialized but non-JAR Java class.

The single-file javac bridge used to reject valid cross-source references before
the producer's generated class was placed into a JAR, even with a graph port.
These tests compile the real Java bridge against companion source files.
"""
from __future__ import annotations

import hashlib

import pytest

from minecraft_mod_ai.integrity_validators import (
    _verified_java_companions, validate_side,
)
from minecraft_mod_ai.javac_bridge import analyze_java


REGISTRY = """package demo;
public class RegistryOwner {
    public static final EntityOwner INSTANCE = new EntityOwner();
}
"""
ENTITY = "package demo; public class EntityOwner { public EntityOwner() {} }"


def test_javac_requires_real_companion_for_generated_java_owner():
    with pytest.raises(ValueError, match="JAVA_ANALYSIS_FAILED"):
        analyze_java(REGISTRY, java_version="17", filename="RegistryOwner.java")
    rows = analyze_java(
        REGISTRY, java_version="17", filename="RegistryOwner.java",
        additional_sources={"EntityOwner.java": ENTITY},
    )
    assert rows
    assert any(row["owner"] == "demo/EntityOwner" for row in rows)
    assert all(row["line"] <= 4 for row in rows)


def test_actual_admitted_companion_source_resolves_side_and_preserves_sha(tmp_path):
    relative = "src/main/java/demo/EntityOwner.java"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(ENTITY, encoding="utf-8")
    row = {"path": relative, "sha256": hashlib.sha256(ENTITY.encode()).hexdigest(),
           "authority": "materialized_prerequisite_source"}
    checked = _verified_java_companions(project_root=tmp_path, source_context=[row])
    assert checked == {relative: ENTITY}
    receipt = validate_side(
        REGISTRY, leaf_id="minecraft/entity/registry", side="COMMON",
        classpath=[], java_version="17", project_root=tmp_path, source_context=[row],
    )
    assert receipt["status"] == "PASS"
    path.write_text(ENTITY + " // altered", encoding="utf-8")
    with pytest.raises(ValueError, match="JAVA_COMPANION_SOURCE_CHANGED"):
        validate_side(
            REGISTRY, leaf_id="minecraft/entity/registry", side="COMMON",
            classpath=[], java_version="17", project_root=tmp_path,
            source_context=[row],
        )


def test_unapproved_and_duplicate_companion_sources_are_rejected(tmp_path):
    relative = "src/main/java/demo/EntityOwner.java"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(ENTITY, encoding="utf-8")
    row = {"path": relative, "sha256": hashlib.sha256(ENTITY.encode()).hexdigest(),
           "authority": "materialized_prerequisite_source"}
    with pytest.raises(ValueError, match="JAVA_COMPANION_UNVERIFIED_AUTHORITY"):
        _verified_java_companions(
            project_root=tmp_path, source_context=[{**row, "authority": "model_guess"}],
        )
    with pytest.raises(ValueError, match="JAVA_COMPANION_DUPLICATE_SOURCE_IDENTITY"):
        _verified_java_companions(project_root=tmp_path, source_context=[row, row])


def test_namespaced_companions_with_identical_filenames_resolve_independently(tmp_path):
    sources = {
        "src/main/java/first/Occurrence.java":
            "package first; public class Occurrence { public static int value(){return 1;} }",
        "src/main/java/second/Occurrence.java":
            "package second; public class Occurrence { public static int value(){return 2;} }",
    }
    records = []
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        records.append({"path": relative, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "authority": "materialized_prerequisite_source"})
    checked = _verified_java_companions(project_root=tmp_path, source_context=records)
    rows = analyze_java(
        "public class Listener { int value(){return first.Occurrence.value()+second.Occurrence.value();} }",
        java_version="17", filename="Listener.java", additional_sources=checked,
    )
    assert {row["owner"] for row in rows if row["name"] == "value"} == {
        "first/Occurrence", "second/Occurrence"}


@pytest.mark.parametrize("name", ["../Escape.java", "/Escape.java", "C:/Escape.java",
                                      "part\\Escape.java", "part/../Escape.java", "part/X.java?x=1"])
def test_companion_source_identity_cannot_escape_its_namespace(name):
    with pytest.raises(ValueError, match="JAVA_ADDITIONAL_SOURCE_INVALID"):
        analyze_java("public class Listener {}", java_version="17", filename="Listener.java",
                     additional_sources={name: "public class Escape {}"})
