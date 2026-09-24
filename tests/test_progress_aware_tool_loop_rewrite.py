from __future__ import annotations

import importlib.util
import json

import pytest

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
    assert [item["function"]["name"] for item in selected] == [
        "search_code_rag",
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_materialized_atomic_target_reconciles_stale_fresh_authority(tmp_path, newline) -> None:
    from types import SimpleNamespace

    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    source = (
        "package dev.mmm.debugfixture;\n"
        "public final class DebugToken { static final int VALUE = 1; }\n"
    )
    target_file = tmp_path / target
    target_file.parent.mkdir(parents=True)
    source = source.replace("\n", newline)
    target_file.write_text(source, encoding="utf-8", newline="")

    payload = {
        "primary_path": target,
        "writable_paths": [target],
        "reuse_action": "fresh",
    }
    messages = [{"role": "developer", "content": json.dumps(payload)}]
    state = loop.HostRunState()

    assert loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.is_new_file is True

    reconciled = loop._reconcile_materialized_target_from_workspace(
        state,
        SimpleNamespace(workspace_root=str(tmp_path)),
    )

    assert reconciled is not None
    assert reconciled.is_new_file is False
    assert reconciled.evidence_source == "workspace_existing_target"
    assert reconciled.source_body == source
    assert target not in reconciled.creatable_paths
    assert target in state.created_paths
    assert loop._host_target_execution_authority(state) is True

    # Rebinding the original host-reserved authority must not resurrect stale
    # create semantics once the exact staged file has been observed.
    assert loop.is_mutation_ready(messages, state) is True
    assert state.mutation_context is not None
    assert state.mutation_context.is_new_file is False
    assert state.mutation_context.source_body == source

    refresh = loop._existing_target_refresh_message(state.mutation_context)
    assert target in refresh["content"]
    assert source.strip() in refresh["content"]


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


def test_native_core_has_no_legacy_runtime_monkey_patch_installers() -> None:
    for module_name in (
        "minecraft_mod_ai.mutation_authority_final_guard",
        "minecraft_mod_ai.planir_mutation_authority_contract",
        "minecraft_mod_ai.repair_mutation_recovery_contract",
    ):
        assert importlib.util.find_spec(module_name) is None

    assert loop._mmm_planir_mutation_authority_v1 is True
    assert loop._mmm_repair_mutation_recovery_v1 is True
    assert loop._mmm_mutation_authority_final_guard_v1 is True
    assert loop._mmm_post_argument_semantic_boundary_v1 is True


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
        SimpleNamespace(_agent_require_fresh_evidence=True),
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

    with pytest.raises(ModelConfigurationError, match="IMPLEMENTATION_EVIDENCE_STALLED"):
        loop.generate_with_tools(
            SimpleNamespace(_agent_require_fresh_evidence=True),
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


def test_fresh_java_retrieval_exhaustion_transitions_to_compile_probe(monkeypatch) -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai import small_model_task_capsule_contract as capsules
    from minecraft_mod_ai.model_adapters import (
        GenerationRequest,
        GenerationResponse,
        ToolCall,
    )

    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    monkeypatch.setattr(
        capsules,
        "current_task_required_gates",
        lambda: ("target_compile",),
    )
    monkeypatch.setattr(capsules, "current_task_reuse_action", lambda: "fresh")

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            names = {item["function"]["name"] for item in request.tools}
            if self.calls == 1:
                assert names == {"search_code_rag"}
                arguments = {"query": "CustomFeature"}
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="ev-1",
                            name="search_code_rag",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments),
                        ),
                    )
                )
            if self.calls == 2:
                assert names == {"search_project_rag"}
                arguments = {"query": "CustomFeature"}
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="ev-2",
                            name="search_project_rag",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments),
                        ),
                    )
                )
            assert self.calls == 3
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
                        raw_arguments=json.dumps(arguments),
                    ),
                )
            )

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, stage, name, _arguments):
            assert stage == "generation"
            self.calls.append(name)
            if name in {"search_code_rag", "search_project_rag"}:
                return {
                    "schema_version": "mmm/code-rag-result-v1",
                    "receipt": {
                        "result_count": 0,
                        "coverage_score": 0.0,
                        "relevance_score": 0.0,
                    },
                    "hits": [],
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
                            "after_sha256": "sha256:" + "1" * 64,
                        }
                    ],
                }
            if name == "target_compile":
                return {
                    "schema_version": "mmm/generation-target-compile-v1",
                    "status": "PASS",
                    "target_path": target,
                    "diagnostics": [],
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
                        "creatable_paths": [target],
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
                    "name": "search_code_rag",
                    "description": "search code",
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
                    "description": "search project",
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
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    runtime = Runtime()
    result = loop.generate_with_tools(
        SimpleNamespace(_agent_require_fresh_evidence=True),
        config=SimpleNamespace(
            adapter="test",
            max_context=32768,
            max_input_tokens=0,
            max_new_tokens=512,
        ),
        adapter=Adapter(),
        request=request,
        runtime=runtime,
        stage="generation",
        role="coder",
    )

    payload = json.loads(result)
    assert "passed generation-time host verification" in payload["summary"]
    assert runtime.calls == [
        "search_code_rag",
        "search_project_rag",
        "apply_source_edit",
        "target_compile",
    ]
