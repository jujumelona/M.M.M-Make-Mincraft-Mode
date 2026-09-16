from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    _CURRENT_AUTHORITY,
    compile_direct_task_mutation_authority,
    install,
)
from minecraft_mod_ai.mutation_authority import MutationAuthorityMode


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


def test_same_active_authority_guards_tool_call_and_final_staged_operation() -> None:
    holder = {}

    class Generator:
        def generate(self, *args, **kwargs):
            active = _CURRENT_AUTHORITY.get()
            assert active is not None
            tool_error = holder["loop"]._mutation_target_error(
                "apply_source_edit",
                {
                    "operation": "create",
                    "path": "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
                },
                SimpleNamespace(target_path="src/main/resources/fabric.mod.json"),
            )
            assert tool_error is None
            self._validate_operations(
                [
                    {
                        "operation": "create",
                        "path": "src/main/java/ai/minecraft/generated/SpaceModeMod.java",
                    }
                ]
            )
            return {"active": active.is_bounded_authored_design}

        def _validate_operations(self, operations):
            return None

    class Loop:
        _SOURCE_EDIT_PATH_KEYS = ("path",)

        @staticmethod
        def _mutation_target_error(tool_name, arguments, context):
            return "MUTATION_TARGET_DRIFT: localized evidence target differs"

        @staticmethod
        def _generate_turn_with_context_recovery(*args, **kwargs):
            return None

        @staticmethod
        def generate_with_tools(*args, **kwargs):
            return "ok"

    custom_module = SimpleNamespace(
        CustomModuleGenerator=Generator,
        CustomModuleGenerationError=RuntimeError,
    )
    loop_module = Loop()
    holder["loop"] = loop_module
    install(
        custom_module_generator_module=custom_module,
        loop_module=loop_module,
    )

    generator = custom_module.CustomModuleGenerator()
    assert generator.generate(module=_authored_module()) == {"active": True}
    assert _CURRENT_AUTHORITY.get() is None
