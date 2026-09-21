from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    compile_direct_task_mutation_authority,
)
from minecraft_mod_ai.mutation_authority import (
    AUTHORED_DESIGN_ROOTS,
    MutationAuthority,
    MutationAuthorityMode,
    canonical_mutation_path,
)


def _authored_module(*, java_package: str = ""):
    config = {
        "authored_plan": {
            "schema_version": "mmm/authored-plan-v1",
            "requested_prompt": "build a complete space mod",
            "text": "preserve the complete authored design",
        }
    }
    if java_package:
        config["authored_java_package"] = java_package
    return SimpleNamespace(
        module_id="task-1",
        kind="custom_java",
        config=config,
    )


def _exact_module():
    anchor = {
        "kind": "symbol",
        "locator": "src/main/java/example/DirectTask.java#DirectTask",
        "status": "host_reserved",
    }
    return SimpleNamespace(
        module_id="direct-task",
        kind="custom_java",
        config={
            "evidence_task": {
                "task_id": "direct-task",
                "owned_anchors": [anchor],
                "production_bindings": [
                    {
                        "task_ref": "direct-task",
                        "reuse_action": "fresh",
                        "owned_anchors": [anchor],
                    }
                ],
            }
        },
    )


def test_authored_scope_is_compiled_from_trusted_host_module_not_messages():
    authority = compile_direct_task_mutation_authority(_authored_module())

    assert authority is not None
    assert authority.task_id == "task-1"
    assert authority.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
    assert authority.mutation_authority.roots == AUTHORED_DESIGN_ROOTS


def test_new_authored_project_narrows_java_roots_to_host_package():
    package = "ai.minecraft.generated.authored_space"
    authority = compile_direct_task_mutation_authority(
        _authored_module(java_package=package)
    )

    assert authority is not None
    assert authority.mutation_authority.roots == (
        "src/main/java/ai/minecraft/generated/authored_space/",
        "src/main/resources/",
        "src/test/java/ai/minecraft/generated/authored_space/",
        "src/gametest/ai/minecraft/generated/authored_space/",
    )
    assert authority.mutation_authority.mutation_error(
        "src/main/java/ai/minecraft/generated/authored_space/ShipSystem.java",
        operation="create_file",
    ) is None
    error = authority.mutation_authority.mutation_error(
        "src/main/java/com/example/starforge/StarForgeMod.java",
        operation="create_file",
    )
    assert error is not None
    assert error.startswith("PATH_OUTSIDE_WRITABLE_SET")


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
    authority = MutationAuthority.bounded_roots(AUTHORED_DESIGN_ROOTS)
    assert authority.mutation_error(path, operation="create_file") is None


@pytest.mark.parametrize(
    "path",
    [
        "build.gradle",
        ".mmm/state.json",
        "../outside.java",
        "/tmp/absolute.java",
        "file:///tmp/absolute.java",
        "https://example.invalid/outside.java",
        "C:/outside.java",
        "src/main/java/../resources/escape.json",
        "src/main/java",
        "src/main/javaevil/NotAllowed.java",
    ],
)
def test_authored_scope_rejects_paths_outside_bounded_roots(path: str):
    authority = MutationAuthority.bounded_roots(AUTHORED_DESIGN_ROOTS)
    assert authority.mutation_error(path, operation="create_file") is not None


def test_authored_scope_rejects_delete_even_inside_safe_root():
    authority = MutationAuthority.bounded_roots(AUTHORED_DESIGN_ROOTS)
    error = authority.mutation_error(
        "src/main/java/example/Old.java",
        operation="delete_file",
    )
    assert error is not None
    assert error.startswith("WRITE_SCOPE_DELETE_FORBIDDEN")


def test_canonical_path_rejects_traversal_absolute_uri_and_drive_paths():
    for path in (
        "../outside.java",
        "/tmp/outside.java",
        "file:///tmp/outside.java",
        "https://example.invalid/outside.java",
        "C:/outside.java",
        "src/main/java/../../outside.java",
    ):
        assert canonical_mutation_path(path) == ""


def test_ordinary_planir_task_remains_exact():
    authority = compile_direct_task_mutation_authority(_exact_module())

    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
    assert authority.mutation_authority.mutation_error(
        "src/main/java/example/DirectTask.java",
        operation="create",
    ) is None
    error = authority.mutation_authority.mutation_error(
        "src/main/java/example/Other.java",
        operation="create",
    )
    assert error is not None
    assert error.startswith("MUTATION_TARGET_DRIFT")
