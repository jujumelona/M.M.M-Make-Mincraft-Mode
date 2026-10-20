"""Regression: registry owner references a materialized but non-JAR Java class.

The single-file javac bridge used to reject valid cross-source references before
the producer's generated class was placed into a JAR, even with a graph port.
These tests compile the real Java bridge against companion source files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

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
    assert checked == {"EntityOwner.java": ENTITY}
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
    with pytest.raises(ValueError, match="JAVA_COMPANION_DUPLICATE_SIMPLE_NAME"):
        _verified_java_companions(project_root=tmp_path, source_context=[row, row])
