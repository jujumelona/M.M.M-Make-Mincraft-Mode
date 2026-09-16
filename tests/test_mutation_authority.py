from __future__ import annotations

import pytest

from minecraft_mod_ai.mutation_authority import (
    AUTHORED_DESIGN_ROOTS,
    MutationAuthority,
    MutationAuthorityError,
    canonical_mutation_path,
)


def test_exact_authority_allows_only_host_paths() -> None:
    authority = MutationAuthority.exact(
        ["src/main/resources/fabric.mod.json"],
        task_id="direct-task",
    )

    assert authority.mutation_error(
        "src/main/resources/fabric.mod.json",
        operation="edit",
    ) is None
    error = authority.mutation_error(
        "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
        operation="create",
    )
    assert error is not None
    assert error.startswith("MUTATION_TARGET_DRIFT:")


def test_authored_design_bounded_authority_allows_generated_java() -> None:
    authority = MutationAuthority.bounded_roots(
        AUTHORED_DESIGN_ROOTS,
        task_id="authored-design",
    )

    assert authority.mutation_error(
        "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
        operation="create",
    ) is None
    assert authority.mutation_error(
        "src/main/resources/assets/example/lang/en_us.json",
        operation="edit",
    ) is None


def test_bounded_authority_blocks_delete_and_host_owned_paths() -> None:
    authority = MutationAuthority.bounded_roots(AUTHORED_DESIGN_ROOTS)

    delete_error = authority.mutation_error(
        "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
        operation="delete",
    )
    assert delete_error is not None
    assert delete_error.startswith("WRITE_SCOPE_DELETE_FORBIDDEN:")

    for path in (
        ".mmm/state.json",
        "build.gradle",
        "config/model_registry.yaml",
        "gradle/wrapper/gradle-wrapper.properties",
    ):
        assert authority.mutation_error(path, operation="edit") is not None


def test_unsafe_path_forms_fail_closed() -> None:
    authority = MutationAuthority.bounded_roots(AUTHORED_DESIGN_ROOTS)

    for path in (
        "../src/main/java/Escape.java",
        "/tmp/Escape.java",
        "file:///tmp/Escape.java",
        "https://example.invalid/Escape.java",
        "C:\\tmp\\Escape.java",
        "src/main/java/../../Escape.java",
    ):
        assert canonical_mutation_path(path) == ""
        error = authority.mutation_error(path, operation="edit")
        assert error is not None
        assert error.startswith("PATH_OUTSIDE_WRITABLE_SET:")


def test_bounded_roots_cannot_be_widened() -> None:
    with pytest.raises(MutationAuthorityError, match="MUTATION_AUTHORITY_ROOT_FORBIDDEN"):
        MutationAuthority.bounded_roots(("src/",))
