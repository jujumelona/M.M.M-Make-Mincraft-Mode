from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.authored_design_mutation_scope import (
    AuthoredMutationScope,
    _canonical_scoped_path,
    _resolve_authored_scope,
    _scoped_mutation_error,
)


def _messages(*, task_id: str = "task-1", phase: str = "implement_authored_design"):
    payload = {
        "phase": phase,
        "workspace_project_root": ".",
        "module": {
            "module_id": task_id,
            "authored_plan": {"requested_prompt": "build a complete space mod"},
        },
    }
    return ({"role": "user", "content": json.dumps(payload)},)


def test_authored_scope_requires_matching_host_authority():
    authority = SimpleNamespace(task_id="task-1")
    scope = _resolve_authored_scope(
        _messages(), authority=authority, stage="generation", role="coder"
    )
    assert scope == AuthoredMutationScope(task_id="task-1")

    assert _resolve_authored_scope(
        _messages(), authority=None, stage="generation", role="coder"
    ) is None
    assert _resolve_authored_scope(
        _messages(), authority=SimpleNamespace(task_id="other"), stage="generation", role="coder"
    ) is None


def test_ordinary_implementation_phase_does_not_receive_broad_scope():
    scope = _resolve_authored_scope(
        _messages(phase="implement_module"),
        authority=SimpleNamespace(task_id="task-1"),
        stage="generation",
        role="coder",
    )
    assert scope is None


@pytest.mark.parametrize(
    "path",
    [
        "src/main/java/example/GalacticFrontierMod.java",
        "src/main/resources/assets/galactic_frontier/lang/en_us.json",
        "src/test/java/example/GalacticFrontierTest.java",
        "src/gametest/example/GalacticFrontierGameTest.java",
    ],
)
def test_authored_scope_allows_only_generated_source_roots(path: str):
    assert _canonical_scoped_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "build.gradle",
        ".mmm/state.json",
        "../outside.java",
        "/tmp/absolute.java",
        "C:/outside.java",
        "src/main/java/../resources/escape.json",
        "src/main/java",
        "src/main/javaevil/NotAllowed.java",
    ],
)
def test_authored_scope_rejects_paths_outside_bounded_roots(path: str):
    assert _canonical_scoped_path(path) == ""


def test_original_galactic_frontier_create_is_authorized():
    scope = AuthoredMutationScope(task_id="task-1")
    error = _scoped_mutation_error(
        "apply_source_edit",
        {
            "operation": "create_file",
            "path": "src/main/java/ai/minecraft/generated/authored_7bf498f5bbba/GalacticFrontierMod.java",
            "content": "package ai.minecraft.generated.authored_7bf498f5bbba;",
        },
        path_keys=("path", "file_path", "target_path"),
        scope=scope,
    )
    assert error is None


def test_authored_scope_rejects_delete_even_inside_safe_root():
    scope = AuthoredMutationScope(task_id="task-1")
    error = _scoped_mutation_error(
        "apply_source_edit",
        {"operation": "delete_file", "path": "src/main/java/example/Old.java"},
        path_keys=("path",),
        scope=scope,
    )
    assert error is not None
    assert error.startswith("WRITE_SCOPE_DELETE_FORBIDDEN")


def test_authored_scope_reports_outside_write_scope():
    scope = AuthoredMutationScope(task_id="task-1")
    error = _scoped_mutation_error(
        "apply_source_edit",
        {"operation": "create_file", "path": "build.gradle"},
        path_keys=("path",),
        scope=scope,
    )
    assert error is not None
    assert error.startswith("PATH_OUTSIDE_WRITABLE_SET")
