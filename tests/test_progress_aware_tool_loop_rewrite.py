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
