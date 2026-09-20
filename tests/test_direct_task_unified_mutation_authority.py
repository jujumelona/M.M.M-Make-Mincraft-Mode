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


def test_authored_bounded_authority_enters_act_without_localization_rag() -> None:
    import json

    from minecraft_mod_ai.model_adapters import (
        GenerationRequest,
        GenerationResponse,
        ToolCall,
    )

    module = _authored_module()
    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.BOUNDED_ROOTS

    target = "src/main/java/ai/minecraft/generated/SpaceModeMod.java"

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            assert self.calls == 1
            names = {item["function"]["name"] for item in request.tools}
            assert names == {"apply_source_edit"}
            assert request.tool_choice == {
                "type": "function",
                "function": {"name": "apply_source_edit"},
            }
            contents = [str(message.get("content") or "") for message in request.messages]
            assert not any(
                value.startswith("MMM reviewed Skill/tool/Minecraft-MCP routing context:")
                for value in contents
            )
            assert any(
                "saved authored design has host-owned bounded-root write authority" in value
                and "src/main/java/" in value
                and "src/main/resources/" in value
                for value in contents
            )
            arguments = {
                "operation": "create_file",
                "path": target,
                "content": (
                    "package ai.minecraft.generated; "
                    "public final class SpaceModeMod {}\n"
                ),
            }
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="authored-edit-1",
                        name="apply_source_edit",
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments, separators=(",", ":")),
                    ),
                )
            )

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, stage, name, arguments):
            assert stage == "generation"
            self.calls.append(name)
            if name == "apply_source_edit":
                assert arguments["path"] == target
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "create",
                            "path": target,
                            "before_sha256": None,
                            "after_sha256": "sha256:" + "7" * 64,
                        }
                    ],
                }
            if name == "java_diagnostics":
                return {
                    "schema_version": "mmm/java-diagnostics-v3",
                    "status": "PASS",
                    "available": True,
                    "complete": True,
                    "session_id": "session",
                    "model_id": "model",
                    "files_opened": 1,
                    "error_count": 0,
                    "warning_count": 0,
                    "diagnostics": {},
                }
            raise AssertionError(name)

    request = GenerationRequest(
        messages=(
            {
                "role": "system",
                "content": (
                    "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
                    + "x" * 20000
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_authored_design",
                        "task": "Implement the saved authored design.",
                        "module": module.config,
                    },
                    separators=(",", ":"),
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "search reviewed project code",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_project_rag",
                    "description": "search version-pinned evidence",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "description": "edit one source/resource file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["create_file", "replace_exact"],
                            },
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["operation", "path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "java_diagnostics",
                    "description": "verify Java",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    adapter = Adapter()
    runtime = Runtime()
    authority_token = CURRENT_MUTATION_AUTHORITY.set(authority.mutation_authority)
    envelope_token = _CURRENT_AUTHORITY.set(authority)
    try:
        result = tool_loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=False),
            config=SimpleNamespace(
                adapter="test",
                max_context=32768,
                max_input_tokens=0,
                max_new_tokens=512,
            ),
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )
    finally:
        _CURRENT_AUTHORITY.reset(envelope_token)
        CURRENT_MUTATION_AUTHORITY.reset(authority_token)

    assert json.loads(result)["summary"]
    assert adapter.calls == 1
    assert runtime.calls == ["apply_source_edit", "java_diagnostics"]
