from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.direct_task_mutation_authority_contract import (
    _CURRENT_AUTHORITY,
    DirectTaskMutationAuthorityError,
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


def _existing_exact_module():
    anchor = {
        "kind": "symbol",
        "locator": (
            "src/main/java/ai/minecraft/generated/authored_demo/"
            "AuthoredFeature001.java#AuthoredFeature001"
        ),
        "status": "existing",
        "ownership": "host_exact_authored_lowering",
        "module_id": "authored_feature_001",
        "source_set": "main",
    }
    return SimpleNamespace(
        module_id="authored_feature_001",
        kind="custom_java",
        config={
            "evidence_task": {
                "task_id": "authored_feature_001",
                "semantic_outcome": "implement one authored feature",
                "implementation_obligations": ["implement this exact feature only"],
                "engineering_worksheet": {"objective": "one exact authored feature"},
                "owned_anchors": [anchor],
                "production_bindings": [
                    {
                        "task_ref": "authored_feature_001",
                        "reuse_action": "fresh",
                        "owned_anchors": [anchor],
                    }
                ],
                "required_gates": ["target_compile"],
            }
        },
        required_gates=("target_compile",),
    )


def test_raw_authored_module_is_rejected_before_model_decode() -> None:
    with pytest.raises(
        DirectTaskMutationAuthorityError,
        match="AUTHORED_LOCALIZATION_REQUIRED",
    ):
        compile_direct_task_mutation_authority(_authored_module())
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


def test_source_owned_progress_loop_uses_same_active_exact_authority() -> None:
    module = _existing_exact_module()
    module.config["evidence_task"]["target_cell"] = {
        "minecraft_version": "26.2",
        "loader": "fabric",
        "mappings": "",
        "java_version": "25",
    }

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
            execution_feedback=None,
        ):
            del project_root, research_modules, minecraft_version, loader, mappings
            del execution_feedback
            active = _CURRENT_AUTHORITY.get()
            assert active is not None
            assert active.mutation_authority.mode is MutationAuthorityMode.EXACT
            assert CURRENT_MUTATION_AUTHORITY.get() is active.mutation_authority
            assert current_mutation_error(
                active.primary_path,
                operation="replace",
            ) is None
            assert current_mutation_error(
                "src/main/java/ai/minecraft/generated/Other.java",
                operation="replace",
            ) is not None
            return {"active": "exact", "path": active.primary_path}

        def _validate_operations(self, operations):
            return None

    generator = Generator()
    result = generator.generate(".", module=module)
    assert result["active"] == "exact"
    assert result["path"].endswith("AuthoredFeature001.java")
    assert _CURRENT_AUTHORITY.get() is None
    assert CURRENT_MUTATION_AUTHORITY.get() is None
def test_fresh_authored_exact_existing_task_first_turn_is_new_only_act(tmp_path) -> None:
    import json

    from minecraft_mod_ai.model_adapters import (
        GenerationRequest,
        GenerationResponse,
        ToolCall,
    )
    from minecraft_mod_ai.small_model_task_capsule_contract import (
        _CURRENT_CAPSULE,
        compile_task_capsule,
    )

    module = _existing_exact_module()
    authority = compile_direct_task_mutation_authority(module)
    capsule = compile_task_capsule(module)
    assert authority is not None
    assert capsule is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
    assert capsule.creatable_paths == ()

    target = capsule.primary_path
    source = (
        "package ai.minecraft.generated.authored_demo;\n"
        "public final class AuthoredFeature001 {\n"
        "    private AuthoredFeature001() {}\n"
        "    public static void initialize() {\n"
        "        // MMM_AUTHORED_FEATURE_BODY_001\n"
        "    }\n"
        "}\n"
    )
    updated = source.replace(
        "// MMM_AUTHORED_FEATURE_BODY_001",
        'System.out.println("ready");',
    )
    target_file = tmp_path / target
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(source, encoding="utf-8", newline="")

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            assert self.calls == 1
            assert request.tool_choice == {
                "type": "function",
                "function": {"name": "apply_source_edit"},
            }
            names = {item["function"]["name"] for item in request.tools}
            assert names == {"apply_source_edit"}
            schema = request.tools[0]["function"]["parameters"]
            assert set(schema["properties"]) == {"new"}
            assert schema["required"] == ["new"]
            assert schema["additionalProperties"] is False
            assert not any(
                name.startswith("search_") for name in names
            )
            arguments = {"new": updated}
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="exact-existing-edit",
                        name="apply_source_edit",
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments, separators=(",", ":")),
                    ),
                )
            )

    class Runtime:
        def __init__(self) -> None:
            self.workspace_root = str(tmp_path)
            self.calls: list[str] = []

        def call(self, stage, name, arguments):
            assert stage == "generation"
            self.calls.append(name)
            if name == "apply_source_edit":
                assert arguments == {
                    "operation": "replace_exact",
                    "path": target,
                    "old": source,
                    "new": updated,
                    "count": 1,
                }
                target_file.write_text(updated, encoding="utf-8", newline="")
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "replace_exact",
                            "path": target,
                            "before_sha256": "sha256:" + "1" * 64,
                            "after_sha256": "sha256:" + "2" * 64,
                        }
                    ],
                }
            if name == "target_compile":
                assert arguments == {"target_path": target}
                return {
                    "status": "PASS",
                    "reason": "exact authored task compiles",
                    "diagnostics": [],
                }
            raise AssertionError(name)

    request = GenerationRequest(
        messages=(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_module",
                        "module": {
                            "module_id": module.module_id,
                            "kind": module.kind,
                            "evidence_task": module.config["evidence_task"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "search workspace",
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
                    "description": "edit source",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["create_file", "replace_exact", "insert_after"],
                            },
                            "path": {"type": "string"},
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "count": {"type": "integer"},
                            "content": {"type": "string"},
                        },
                        "required": ["operation", "path"],
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
    capsule_token = _CURRENT_CAPSULE.set(capsule)
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
        _CURRENT_CAPSULE.reset(capsule_token)
        _CURRENT_AUTHORITY.reset(envelope_token)
        CURRENT_MUTATION_AUTHORITY.reset(authority_token)

    assert json.loads(result)["summary"]
    assert adapter.calls == 1
    assert runtime.calls == ["apply_source_edit", "target_compile"]
    assert target_file.read_text(encoding="utf-8") == updated


def test_raw_authored_cannot_enter_act_without_localization() -> None:
    with pytest.raises(
        DirectTaskMutationAuthorityError,
        match="AUTHORED_LOCALIZATION_REQUIRED",
    ):
        compile_direct_task_mutation_authority(_authored_module())
def test_localized_authored_task_keeps_project_build_gate_when_jdt_is_unavailable() -> None:
    module = _existing_exact_module()
    authority = compile_direct_task_mutation_authority(module)

    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
    assert "target_compile" in module.required_gates
    # JDT availability is a verifier concern; exact mutation authority must not widen
    # just because a later verifier falls back to the project build gate.
    assert authority.mutation_authority.mutation_error(
        authority.primary_path,
        operation="replace",
    ) is None
    drift = authority.mutation_authority.mutation_error(
        "src/main/java/ai/minecraft/generated/Fallback.java",
        operation="replace",
    )
    assert drift is not None
    assert drift.startswith("MUTATION_TARGET_DRIFT")
def test_later_authored_fragment_cannot_widen_frozen_target_set() -> None:
    module = _existing_exact_module()
    authority = compile_direct_task_mutation_authority(module)

    assert authority is not None
    assert authority.mutation_authority.mode is MutationAuthorityMode.EXACT
    frozen = authority.primary_path
    assert authority.mutation_authority.mutation_error(
        frozen,
        operation="replace",
    ) is None
    for candidate in (
        "src/main/java/com/example/spacemode/TradeSystem.java",
        "src/main/resources/assets/spacemode/lang/en_us.json",
    ):
        error = authority.mutation_authority.mutation_error(
            candidate,
            operation="create_file",
        )
        assert error is not None
        assert error.startswith("MUTATION_TARGET_DRIFT")
