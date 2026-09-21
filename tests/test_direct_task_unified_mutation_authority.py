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
            execution_feedback=None,
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
    target_file.write_text(source, encoding="utf-8")

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
                target_file.write_text(updated, encoding="utf-8")
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
                        "host_grounding": {
                            "schema_version": "mmm/host-owned-coder-grounding-v1",
                            "policy": {
                                "resolved_before_first_coder_decode": True,
                                "baseline_grounding_owned_by_host": True,
                                "baseline_grounding_optional_for_model": False,
                                "model_tool_choice_required_for_baseline": False,
                            },
                            "evidence_bindings": {
                                "project_exact_rag": {
                                    "receipt": {
                                        "observation_count": 1,
                                        "project_sha256": "sha256:" + "1" * 64,
                                        "observations_sha256": "sha256:" + "2" * 64,
                                    }
                                }
                            },
                        },
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


def test_authored_jdt_unavailable_defers_to_project_build() -> None:
    import json

    from minecraft_mod_ai.model_adapters import (
        GenerationRequest,
        GenerationResponse,
        ToolCall,
    )

    module = _authored_module()
    authority = compile_direct_task_mutation_authority(module)
    assert authority is not None
    target = "src/main/java/ai/minecraft/generated/SpaceModeMod.java"

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            assert self.calls == 1
            assert {item["function"]["name"] for item in request.tools} == {
                "apply_source_edit"
            }
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
                        id="authored-edit-unavailable",
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
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "create",
                            "path": target,
                            "before_sha256": None,
                            "after_sha256": "sha256:" + "8" * 64,
                        }
                    ],
                }
            if name == "java_diagnostics":
                return {
                    "schema_version": "mmm/java-diagnostics-v3",
                    "status": "UNAVAILABLE",
                    "available": False,
                    "complete": False,
                    "skipped": True,
                    "error_count": 0,
                    "warning_count": 0,
                    "diagnostics": [
                        {
                            "severity": 1,
                            "code": "JDT_DIAGNOSTICS_UNAVAILABLE",
                            "source": "jdt_core",
                            "message": "Loom dependency download failed",
                        }
                    ],
                }
            raise AssertionError(name)

    request = GenerationRequest(
        messages=(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_authored_design",
                        "task": "Implement the saved authored design.",
                        "module": module.config,
                        "host_grounding": {
                            "schema_version": "mmm/host-owned-coder-grounding-v1",
                            "policy": {
                                "resolved_before_first_coder_decode": True,
                                "baseline_grounding_owned_by_host": True,
                                "baseline_grounding_optional_for_model": False,
                                "model_tool_choice_required_for_baseline": False,
                            },
                            "evidence_bindings": {
                                "project_exact_rag": {
                                    "receipt": {
                                        "observation_count": 1,
                                        "project_sha256": "sha256:" + "1" * 64,
                                        "observations_sha256": "sha256:" + "2" * 64,
                                    }
                                }
                            },
                        },
                    },
                    separators=(",", ":"),
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "description": "edit one source/resource file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
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

    summary = json.loads(result)["summary"]
    assert "project build" in summary
    assert adapter.calls == 1
    assert runtime.calls == ["apply_source_edit", "java_diagnostics"]
    assert tool_loop.current_generation_verification_receipt() is None



def test_later_authored_fragment_refreshes_workspace_before_act() -> None:
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

    new_target = "src/main/java/com/example/spacemode/TradeSystem.java"

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            names = {item["function"]["name"] for item in request.tools}
            if self.calls == 1:
                assert names == {"search_code_rag"}
                assert request.tool_choice == {
                    "type": "function",
                    "function": {"name": "search_code_rag"},
                }
                arguments = {"query": "current generated package ModInitializer entrypoint"}
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="refresh-workspace",
                            name="search_code_rag",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments, separators=(",", ":")),
                        ),
                    )
                )
            if self.calls == 2:
                assert names == {"apply_source_edit"}
                assert request.tool_choice == {
                    "type": "function",
                    "function": {"name": "apply_source_edit"},
                }
                rendered = "\n".join(
                    str(message.get("content") or "") for message in request.messages
                )
                assert "SpaceModeModule" in rendered
                arguments = {
                    "operation": "create_file",
                    "path": new_target,
                    "content": (
                        "package com.example.spacemode; "
                        "public final class TradeSystem {}\n"
                    ),
                }
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="authored-fragment-edit",
                            name="apply_source_edit",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments, separators=(",", ":")),
                        ),
                    )
                )
            raise AssertionError(f"unexpected coder call {self.calls}")

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, stage, name, arguments):
            assert stage == "generation"
            self.calls.append(name)
            if name == "search_code_rag":
                return {
                    "parsed_text": None,
                    "resources": [],
                    "structured_content": {
                        "schema_version": "mmm/code-rag-result-v1",
                        "query": str(arguments.get("query") or ""),
                        "hits": [
                            {
                                "path": "src/main/java/com/example/spacemode/SpaceModeModule.java",
                                "source_path": "src/main/java/com/example/spacemode/SpaceModeModule.java",
                                "text": (
                                    "package com.example.spacemode; "
                                    "public final class SpaceModeModule {}"
                                ),
                            }
                        ],
                        "receipt": {
                            "status": "FOUND",
                            "result_count": 1,
                        },
                    },
                    "text": [],
                }
            if name == "apply_source_edit":
                assert arguments["path"] == new_target
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "create",
                            "path": new_target,
                            "before_sha256": None,
                            "after_sha256": "sha256:" + "9" * 64,
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
                    "files_opened": 2,
                    "error_count": 0,
                    "warning_count": 0,
                    "diagnostics": {},
                }
            raise AssertionError(name)

    request_payload = {
        "phase": "implement_authored_design",
        "task": "Implement the next saved authored-design fragment.",
        "module": module.config,
        "initial_exact_source_context": {
            "mode": "retrieve_current_authored_fragment_with_tools",
            "reason": "earlier authored fragments changed the staged workspace",
        },
        "authored_execution": {
            "schema_version": "mmm/authored-plan-fragment-v1",
            "fragment_index": 2,
            "fragment_count": 3,
            "source_text_sha256": "sha256:" + "a" * 64,
        },
        "host_grounding": {
            "schema_version": "mmm/host-owned-coder-grounding-v1",
            "policy": {
                "resolved_before_first_coder_decode": True,
                "baseline_grounding_owned_by_host": True,
                "baseline_grounding_optional_for_model": False,
                "model_tool_choice_required_for_baseline": False,
            },
            "evidence_bindings": {
                "project_exact_rag": {
                    "receipt": {
                        "observation_count": 1,
                        "project_sha256": "sha256:" + "1" * 64,
                        "observations_sha256": "sha256:" + "2" * 64,
                    }
                }
            },
        },
    }
    request = GenerationRequest(
        messages=(
            {
                "role": "user",
                "content": json.dumps(request_payload, separators=(",", ":")),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "search current staged project code",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
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
                            "operation": {"type": "string"},
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
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
    assert adapter.calls == 2
    assert runtime.calls == [
        "search_code_rag",
        "apply_source_edit",
        "java_diagnostics",
    ]
