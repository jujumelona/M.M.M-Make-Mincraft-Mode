"""Materialized Java source takes precedence over old local output bytecode."""
from __future__ import annotations

from minecraft_mod_ai.construction_runtime import _reconcile_project_compiled_sources


def test_local_compiled_class_and_members_replaced_by_current_source(tmp_path):
    (tmp_path / "build/classes/java/main").mkdir(parents=True)
    local = str(tmp_path / "build" / "classes" / "java" / "main")
    external = str(tmp_path / "dependencies" / "external.jar")
    original = {
        "example.AuthoredProgram": {
            "kind": "CLASS", "owner": "example.AuthoredProgram", "source_path": local,
        },
        "example.AuthoredProgram#old()V": {
            "kind": "METHOD", "owner": "example.AuthoredProgram", "source_path": local,
        },
        "example.Other": {
            "kind": "CLASS", "owner": "example.Other", "source_path": local,
        },
        "external.API": {
            "kind": "CLASS", "owner": "external.API", "source_path": external,
        },
    }
    pages = {"src/main/java/example/AuthoredProgram.java": [
        {"kind": "class", "owner": "example.AuthoredProgram"},
    ]}
    result = _reconcile_project_compiled_sources(original, pages, tmp_path)
    assert "example.AuthoredProgram" not in result
    assert "example.AuthoredProgram#old()V" not in result
    assert "example.Other" in result
    assert "external.API" in result
    assert len(original) == 4  # cached symbol inventory must not be mutated


def test_external_jar_class_collision_remains_authoritative(tmp_path):
    external = str(tmp_path / "deps" / "third-party.jar")
    original = {"external.API": {
        "kind": "CLASS", "owner": "external.API", "source_path": external,
    }}
    pages = {"src/main/java/external/API.java": [
        {"kind": "class", "owner": "external.API"},
    ]}
    result = _reconcile_project_compiled_sources(original, pages, tmp_path)
    assert result == original  # genuine dependency collisions must fail downstream
