from __future__ import annotations

import json

from minecraft_mod_ai import progress_aware_tool_loop as loop


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def test_fresh_host_reserved_target_is_ready_without_searching_its_own_filename() -> None:
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    payload = {
        "primary_path": path,
        "writable_paths": [path],
        "reuse_action": "fresh",
    }
    messages = [{"role": "developer", "content": json.dumps(payload)}]
    state = loop.HostRunState()

    assert loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.target_path == path
    assert state.mutation_context.is_new_file is True
    assert state.mutation_context.localization_stage is loop.LocalizationStage.READY

    selected = loop._filter_tools_for_phase(
        (_schema("search_code_rag"), _schema("search_project_rag")),
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=state.mutation_context,
        attempted_sources=(),
        semantic_retrieval_choice=True,
    )
    assert [item["function"]["name"] for item in selected] == ["search_project_rag"]


def test_recovery_frontier_excludes_unrelated_ecosystem_discovery() -> None:
    tools = tuple(
        _schema(name)
        for name in (
            "search_code_rag",
            "search_project_rag",
            "java_workspace_symbols",
            "discover_ecosystem_resources",
            "inspect_github_repository",
            "inspect_modrinth_project",
        )
    )
    selected = loop._filter_tools_for_phase(
        tools,
        loop.LoopPhase.RECOVER,
        "coder",
        mutation_context=loop.TargetMutationContext(
            target_path="src/main/java/dev/mmm/Foo.java",
            target_symbol="Foo",
            source_body="public final class Foo {}",
            writable_paths=("src/main/java/dev/mmm/Foo.java",),
            target_pinned=True,
        ),
        attempted_sources=(),
    )
    names = {item["function"]["name"] for item in selected}
    assert names <= {"search_code_rag", "search_project_rag", "java_workspace_symbols"}
    assert "discover_ecosystem_resources" not in names
    assert "inspect_github_repository" not in names
    assert "inspect_modrinth_project" not in names


def test_repeated_unchanged_source_edit_hits_semantic_fixed_point() -> None:
    state = loop.HostRunState(
        phase=loop.LoopPhase.ACT,
        mutation_context=loop.TargetMutationContext(
            target_path="src/main/java/dev/mmm/Foo.java",
            target_symbol="Foo",
            source_body="public final class Foo {}",
            writable_paths=("src/main/java/dev/mmm/Foo.java",),
            target_pinned=True,
        ),
    )
    arguments = {
        "operation": "replace_exact",
        "path": "src/main/java/dev/mmm/Foo.java",
        "old": "Foo",
        "new": "Foo",
    }
    payload = {"ok": True, "result": {"status": "APPLIED"}}

    assert state.record_mutation("apply_source_edit", arguments, payload) is False
    assert state.semantic_fixed_point is False
    assert state.record_mutation("apply_source_edit", arguments, payload) is False
    assert state.semantic_fixed_point is True


def test_native_core_blocks_legacy_runtime_monkey_patch_installers() -> None:
    from minecraft_mod_ai.mutation_authority_final_guard import install as install_final_guard
    from minecraft_mod_ai.planir_mutation_authority_contract import install as install_planir
    from minecraft_mod_ai.repair_mutation_recovery_contract import install as install_repair

    original_generate = loop.generate_with_tools
    original_context = loop.TargetMutationContext
    original_ready = loop.is_mutation_ready
    original_turn = loop._generate_turn_with_context_recovery

    install_planir(loop)
    install_repair(loop)
    install_final_guard(loop)

    assert loop.generate_with_tools is original_generate
    assert loop.TargetMutationContext is original_context
    assert loop.is_mutation_ready is original_ready
    assert loop._generate_turn_with_context_recovery is original_turn


def test_failed_external_mcp_route_is_consumed_for_recovery_frontier() -> None:
    state = loop.HostRunState(
        phase=loop.LoopPhase.RECOVER,
        mutation_context=loop.TargetMutationContext(
            target_path="src/main/java/dev/mmm/Foo.java",
            target_symbol="Foo",
            writable_paths=("src/main/java/dev/mmm/Foo.java",),
            target_pinned=True,
        ),
    )
    state.record_source_attempt("external_mcp_call", {"capability": "read_file"})

    selected = loop._filter_tools_for_phase(
        (
            _schema("search_code_rag"),
            _schema("search_project_rag"),
            _schema("java_workspace_symbols"),
            _schema("external_mcp_call"),
            _schema("inspect_modrinth_project"),
        ),
        loop.LoopPhase.RECOVER,
        "coder",
        mutation_context=state.mutation_context,
        attempted_sources=state.attempted_sources,
    )
    names = {item["function"]["name"] for item in selected}
    assert "external_mcp_call" not in names



def test_fresh_java_write_authority_does_not_replace_api_evidence() -> None:
    path = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    state = loop.HostRunState(
        mutation_context=loop.TargetMutationContext(
            target_path=path,
            target_symbol="DebugToken",
            is_new_file=True,
            evidence_source="evidence_fresh_owned_anchor",
            writable_paths=(path,),
            creatable_paths=(path,),
            target_pinned=True,
        )
    )

    assert loop._host_target_execution_authority(state) is True
    assert loop._target_evidence_ready(
        state,
        require_rag=True,
        fresh_java_target=True,
    ) is False

    assert state.record_evidence(
        {
            "receipt": {
                "result_count": 1,
                "coverage_score": 1.0,
                "relevance_score": 1.0,
            },
            "hits": [
                {
                    "path": "src/main/java/dev/mmm/Example.java",
                    "text": (
                        "package dev.mmm; import net.fabricmc.api.ModInitializer; "
                        "public final class Example {}"
                    ),
                }
            ],
        },
        usable=True,
    ) is True
    assert loop._target_evidence_ready(
        state,
        require_rag=True,
        fresh_java_target=True,
    ) is True


def test_fresh_java_requires_reviewed_evidence_before_source_mutation() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse, ToolCall

    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"

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
                arguments = {"query": "Fabric item registration example"}
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="evidence-1",
                            name="search_code_rag",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments, separators=(",", ":")),
                        ),
                    )
                )
            if self.calls == 2:
                assert names == {"apply_source_edit"}
                arguments = {
                    "operation": "create_file",
                    "path": target,
                    "content": (
                        "package dev.mmm.debugfixture; "
                        "public final class DebugToken {}\n"
                    ),
                }
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="edit-1",
                            name="apply_source_edit",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments, separators=(",", ":")),
                        ),
                    )
                )
            raise AssertionError("terminal verifier state must not invoke the coder again")

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, stage, name, _arguments):
            assert stage == "generation"
            self.calls.append(name)
            if name == "search_code_rag":
                return {
                    "schema_version": "mmm/code-rag-result-v1",
                    "receipt": {
                        "result_count": 1,
                        "coverage_score": 1.0,
                        "relevance_score": 1.0,
                    },
                    "hits": [
                        {
                            "path": "src/main/java/dev/mmm/Existing.java",
                            "text": (
                                "package dev.mmm; import net.fabricmc.api.ModInitializer; "
                                "public final class Existing {}"
                            ),
                        }
                    ],
                }
            if name == "apply_source_edit":
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "create",
                            "path": target,
                            "before_sha256": None,
                            "after_sha256": "sha256:" + "4" * 64,
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
                "role": "developer",
                "content": json.dumps(
                    {
                        "primary_path": target,
                        "writable_paths": [target],
                        "reuse_action": "fresh",
                        "module": {
                            "config": {
                                "evidence_task": {
                                    "owned_anchors": [
                                        {
                                            "kind": "symbol",
                                            "locator": target + "#DebugToken",
                                            "status": "host_reserved",
                                        }
                                    ]
                                }
                            }
                        },
                    }
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_module",
                        "task": "Implement the approved debug token item.",
                    }
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "search reviewed project/API code evidence",
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
                    "name": "search_project_rag",
                    "description": "search version-pinned primary evidence",
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
                    "description": "edit source",
                    "parameters": {"type": "object", "properties": {}},
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

    result = loop.generate_with_tools(
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

    assert json.loads(result)["summary"]
    assert adapter.calls == 2
    assert runtime.calls == [
        "search_code_rag",
        "apply_source_edit",
        "java_diagnostics",
    ]



def test_fresh_java_without_reviewed_evidence_tool_fails_before_model_mutation() -> None:
    from types import SimpleNamespace

    import pytest

    from minecraft_mod_ai.model_adapters import GenerationRequest, ModelConfigurationError

    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"

    class NeverCalledAdapter:
        def generate_turn(self, _request):
            raise AssertionError("fresh Java must not reach ACT without reviewed evidence")

    request = GenerationRequest(
        messages=(
            {
                "role": "developer",
                "content": json.dumps(
                    {
                        "primary_path": target,
                        "writable_paths": [target],
                        "reuse_action": "fresh",
                    }
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_module",
                        "task": "Implement the approved debug token item.",
                    }
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "description": "edit source",
                    "parameters": {"type": "object", "properties": {}},
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

    with pytest.raises(ModelConfigurationError, match="MUTATION_LOCALIZATION_STALLED"):
        loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=False),
            config=SimpleNamespace(
                adapter="test",
                max_context=32768,
                max_input_tokens=0,
                max_new_tokens=512,
            ),
            adapter=NeverCalledAdapter(),
            request=request,
            runtime=SimpleNamespace(),
            stage="generation",
            role="coder",
        )
