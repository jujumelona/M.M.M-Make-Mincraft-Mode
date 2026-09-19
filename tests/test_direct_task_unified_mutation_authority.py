from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    _CURRENT_AUTHORITY,
    compile_direct_task_mutation_authority,
)
from minecraft_mod_ai.small_model_write_scope_enforcement import generation_authority_scoped
from minecraft_mod_ai.mutation_authority import (
    CURRENT_MUTATION_AUTHORITY,
    MutationAuthorityMode,
    current_mutation_error,
)


def _authored_module():
    return SimpleNamespace(
        module_id="space-mode",
        kind="custom_java",
        config={
            "authored_plan": {
                "schema_version": "mmm/authored-plan-v1",
                "requested_prompt": "make a space mode",
                "text": "space mechanics",
            }
        },
    )


def _exact_module():
    anchor = {
        "kind": "symbol",
        "locator": "src/main/java/ai/minecraft/generated/DirectTask.java#DirectTask",
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


def test_host_authored_module_compiles_bounded_authority_before_model_decode() -> None:
    authority = compile_direct_task_mutation_authority(_authored_module())

    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS
    assert authority.mutation_authority.mutation_error(
        "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
        operation="create",
    ) is None


def test_ordinary_planir_task_stays_exact_and_cannot_follow_localization_drift() -> None:
    authority = compile_direct_task_mutation_authority(_exact_module())

    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
    error = authority.mutation_authority.mutation_error(
        "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
        operation="create",
    )
    assert error is not None
    assert error.startswith("MUTATION_TARGET_DRIFT:")


def test_source_owned_progress_loop_uses_same_active_bounded_authority() -> None:

    class Generator:
        @generation_authority_scoped
        def generate(
            self,
            project_root,
            *,
            module,
            research_modules=(),
            minecraft_version=None,
            loader=None,
            mappings=None,
        ):
            active = _CURRENT_AUTHORITY.get()
            assert active is not None
            assert CURRENT_MUTATION_AUTHORITY.get() is active.mutation_authority
            tool_error = tool_loop._mutation_target_error(
                "apply_source_edit",
                {
                    "operation": "create",
                    "path": "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
                },
                None,
            )
            assert tool_error is None
            assert current_mutation_error(
                "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
                operation="create",
            ) is None
            assert current_mutation_error(
                "build.gradle",
                operation="replace",
            ) is not None
            return {"active": active.is_bounded_authored_design}

        def _validate_operations(self, operations):
            return None

    generator = Generator()
    assert generator.generate(".", module=_authored_module()) == {"active": True}
    assert _CURRENT_AUTHORITY.get() is None
    assert CURRENT_MUTATION_AUTHORITY.get() is None
