from __future__ import annotations

"""REG-007: reject unbound, unowned or non-creatable production source targets."""

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    DirectTaskMutationAuthorityError,
    compile_direct_task_mutation_authority,
)

TASK_ID = "task_creation_slot_regression"
JAVA_PATH = "src/main/java/example/CreationSlot.java"
UNOWNED_PATH = "src/main/java/example/Unapproved.java"


def _anchor(path: str = JAVA_PATH, *, status: str = "host_reserved") -> dict:
    return {
        "kind": "symbol",
        "locator": f"{path}#CreationSlot",
        "ownership": "exclusive",
        "status": status,
        "module_id": TASK_ID,
        "source_set": "main",
    }


def _module(*, binding: bool, primary: str = JAVA_PATH,
            owned: str = JAVA_PATH, status: str = "host_reserved"):
    task = {
        "task_id": TASK_ID,
        "task_sha256": "sha256:" + "a" * 64,
        "owned_anchors": [_anchor(owned, status=status)],
        "production_bindings": (
            [{
                "task_ref": TASK_ID,
                "reuse_action": "fresh",
                "owned_anchors": [_anchor(primary)],
            }]
            if binding
            else []
        ),
    }
    return SimpleNamespace(
        module_id=TASK_ID,
        kind="custom_java",
        config={"evidence_task": task},
    )


def test_fresh_host_reserved_task_cannot_silently_fall_back_without_binding():
    with pytest.raises(DirectTaskMutationAuthorityError, match="BINDING_MISSING"):
        compile_direct_task_mutation_authority(_module(binding=False))


def test_fresh_production_binding_cannot_create_a_source_outside_owned_paths():
    with pytest.raises(DirectTaskMutationAuthorityError, match="PRIMARY_NOT_OWNED"):
        compile_direct_task_mutation_authority(
            _module(binding=True, primary=UNOWNED_PATH)
        )


def test_fresh_creation_slot_is_exact_and_cannot_authorize_a_different_path():
    authority = compile_direct_task_mutation_authority(_module(binding=True))
    assert authority is not None
    assert authority.primary_path == JAVA_PATH
    assert authority.creatable_paths == (JAVA_PATH,)
    assert authority.mutation_authority.mutation_error(
        JAVA_PATH, operation="create_file"
    ) is None
    assert authority.mutation_authority.mutation_error(
        UNOWNED_PATH, operation="create_file"
    ).startswith("MUTATION_TARGET_DRIFT:")


def test_existing_source_binding_does_not_grant_creation_permission():
    authority = compile_direct_task_mutation_authority(
        _module(binding=True, status="existing")
    )
    assert authority is not None
    assert authority.creatable_paths == ()
