from __future__ import annotations

import inspect

from minecraft_mod_ai import progress_aware_tool_loop as loop


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def test_recover_exposes_grounding_tools_not_mutation_tools() -> None:
    tools = tuple(
        _schema(name)
        for name in (
            "apply_source_edit",
            "search_code_rag",
            "inspect_modrinth_project",
            "external_mcp_capabilities",
            "external_mcp_schema",
            "external_mcp_call",
        )
    )
    selected = loop._filter_tools_for_phase(tools, loop.LoopPhase.RECOVER, "coder")
    names = {item["function"]["name"] for item in selected}
    assert names == {"search_code_rag"}
    assert "apply_source_edit" not in names


def test_recover_marks_version_api_routes_as_evidence() -> None:
    assert {
        "search_code_rag",
        "search_project_rag",
        "inspect_modrinth_project",
        "external_mcp_call",
    } <= loop._RECOVERY_EVIDENCE_TOOLS


def test_verifier_fail_enters_recover_before_another_edit() -> None:
    source = inspect.getsource(loop._generate_with_tools_impl)
    marker = 'if status == "FAIL" and implementation_requires_mutation:'
    assert marker in source
    branch = source.split(marker, 1)[1].split("continue", 1)[0]
    assert "state.phase = LoopPhase.RECOVER" in branch
    assert "state.phase = LoopPhase.ACT" not in branch


def test_recover_requires_a_tool_call_and_returns_to_act_only_after_evidence() -> None:
    source = inspect.getsource(loop._generate_with_tools_impl)
    assert 'elif state.phase == LoopPhase.RECOVER:\n            tool_choice = "required"' in source
    assert 'state.phase in {LoopPhase.OBSERVE, LoopPhase.RECOVER}' in source



def test_verify_to_recover_handoff_bounds_raw_verifier_receipt() -> None:
    raw_receipt = "RAW_JDT_SENTINEL:" + ("x" * 100_000)
    errors = tuple(
        {
            "path": f"src/main/java/demo/Broken{i}.java",
            "line": i + 1,
            "severity": 1,
            "code": "JDT_ERROR",
            "source": "jdt_core",
            "message": "Unresolved symbol " + ("detail " * 200),
        }
        for i in range(20)
    )
    state = loop.HostRunState(
        phase=loop.LoopPhase.RECOVER,
        validation_status="FAIL",
        latest_verifier_tool="java_diagnostics",
        latest_verifier_errors=errors,
        latest_verifier_fingerprint="sha256:test-verifier-fingerprint",
    )
    messages = [
        {"role": "system", "content": "host authority"},
        {"role": "user", "content": "repair the generated source"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "verify-1",
                    "type": "function",
                    "function": {
                        "name": "java_diagnostics",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "verify-1",
            "content": raw_receipt,
        },
    ]

    phase = loop._sync_phase_tool_transcript(
        messages,
        state=state,
        last_prompt_phase=loop.LoopPhase.VERIFY,
        stage="generation",
    )

    assert phase is loop.LoopPhase.RECOVER
    handoff = messages[-1]["content"]
    assert "RAW_JDT_SENTINEL" not in handoff
    assert "mmm/verifier-recovery-handoff-v1" in handoff
    assert "diagnostics_fingerprint" in handoff
    assert "omitted_diagnostic_count" in handoff
    assert "Unresolved symbol" in handoff
    assert len(handoff.encode("utf-8")) < 8 * 1024



def test_observe_to_act_handoff_bounds_successful_rag_payload() -> None:
    raw_sentinel = "RAW_RAG_SENTINEL:" + ("x" * 100_000)
    payload = {
        "ok": True,
        "tool": "search_code_rag",
        "result": {
            "structured_content": {
                "schema_version": "mmm/code-rag-result-v1",
                "hits": [
                    {
                        "path": "src/main/java/demo/SpaceModeMod.java",
                        "source_path": "src/main/java/demo/SpaceModeMod.java",
                        "text": (
                            "package demo; public final class SpaceModeMod {} "
                            + raw_sentinel
                        ),
                    }
                ],
                "receipt": {"status": "FOUND", "result_count": 1},
            }
        },
    }
    state = loop.HostRunState(
        phase=loop.LoopPhase.ACT,
        evidence_fingerprints={"sha256:evidence"},
    )
    messages = [
        {"role": "system", "content": "host authority"},
        {"role": "user", "content": "implement the next authored fragment"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "rag-1",
                    "type": "function",
                    "function": {
                        "name": "search_code_rag",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "rag-1",
            "name": "search_code_rag",
            "content": __import__("json").dumps(payload),
        },
    ]

    phase = loop._sync_phase_tool_transcript(
        messages,
        state=state,
        last_prompt_phase=loop.LoopPhase.OBSERVE,
        stage="generation",
    )

    assert phase is loop.LoopPhase.ACT
    handoff = messages[-1]["content"]
    assert "mmm/phase-tool-observation-v1" in handoff
    assert "src/main/java/demo/SpaceModeMod.java" in handoff
    assert "RAW_RAG_SENTINEL:" in handoff
    assert raw_sentinel not in handoff
    assert "tool_result_fingerprint" in handoff
    assert len(handoff.encode("utf-8")) <= 6 * 1024


def test_phase_handoff_replaces_previous_snapshot_instead_of_accumulating() -> None:
    state = loop.HostRunState(phase=loop.LoopPhase.ACT)
    messages = [
        {"role": "system", "content": "host authority"},
        {"role": "user", "content": "continue"},
        {
            "role": "system",
            "content": "MMM_PHASE_HANDOFF VERIFY->RECOVER\nold snapshot",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "rag-2",
                    "type": "function",
                    "function": {
                        "name": "search_code_rag",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "rag-2",
            "name": "search_code_rag",
            "content": '{"ok":true,"tool":"search_code_rag","result":{"hits":[{"path":"src/main/java/demo/Now.java","text":"class Now {}"}]}}',
        },
    ]

    loop._sync_phase_tool_transcript(
        messages,
        state=state,
        last_prompt_phase=loop.LoopPhase.OBSERVE,
        stage="generation",
    )

    handoffs = [
        message
        for message in messages
        if str(message.get("role") or "") == "system"
        and str(message.get("content") or "").startswith("MMM_PHASE_HANDOFF ")
    ]
    assert len(handoffs) == 1
    assert "old snapshot" not in handoffs[0]["content"]
    assert "Now.java" in handoffs[0]["content"]



def test_verifier_api_failure_uses_shared_repair_evidence_router() -> None:
    route = loop.repair_evidence_route_for_errors(
        (
            {
                "message": (
                    "package net.minecraft.registry does not exist\n"
                    "import net.minecraft.registry.Registry;"
                )
            },
        )
    )
    assert route["route"] == "official_api"
    assert route["small_model_policy"]["retrieval_owner"] == "host"


def test_local_symbol_failure_stays_project_local_with_exact_source_context() -> None:
    source = "package demo; public final class Foo { void run() { localCounter++; } }"
    route = loop.repair_evidence_route_for_errors(
        (
            {
                "path": "src/main/java/demo/Foo.java",
                "message": (
                    "cannot find symbol\n"
                    "symbol: variable localCounter\n"
                    "location: class Foo"
                ),
            },
        ),
        local_source=source,
        target_path="src/main/java/demo/Foo.java",
    )
    assert route["route"] == "project_local"


def test_recover_frontier_is_host_selected_one_route_at_a_time() -> None:
    state = loop.HostRunState(
        phase=loop.LoopPhase.RECOVER,
        repair_evidence_route="official_api",
        mutation_context=loop.TargetMutationContext(
            target_path="src/main/java/dev/mmm/Foo.java",
            source_body="package dev.mmm; public final class Foo {}",
            writable_paths=("src/main/java/dev/mmm/Foo.java",),
            target_pinned=True,
        ),
    )
    selected = loop._filter_tools_for_phase(
        (
            _schema("search_code_rag"),
            _schema("search_project_rag"),
            _schema("java_workspace_symbols"),
            _schema("external_mcp_capabilities"),
        ),
        loop.LoopPhase.RECOVER,
        "coder",
        mutation_context=state.mutation_context,
        attempted_sources=state.attempted_sources,
        repair_evidence_route=state.repair_evidence_route,
    )
    assert [item["function"]["name"] for item in selected] == ["search_project_rag"]


def test_progress_loop_does_not_duplicate_evidence_or_repair_policy() -> None:
    source = inspect.getsource(loop)
    assert "_API_EVIDENCE_DIAGNOSTIC_MARKERS" not in source
    assert "repair_requires_api_evidence" not in source
    assert "_consume_rejected_evidence_fixed_point" not in source


def test_external_mcp_frontier_snapshot_exposes_completed_and_next_capability() -> None:
    from minecraft_mod_ai.external_mcp_recovery_contract import recovery_state_snapshot

    state = loop.HostRunState()
    state._external_mcp_capabilities_seen = True
    state._external_mcp_recovery_capabilities = (
        "source_search",
        "official_mod_docs",
        "mapping_resolution",
        "registry_lookup",
    )
    state._external_mcp_completed_capabilities = {
        "source_search",
        "official_mod_docs",
        "mapping_resolution",
    }
    state._external_mcp_schema_capability = ""
    snapshot = recovery_state_snapshot(state, "official_api")
    assert snapshot["completed_capabilities"] == [
        "mapping_resolution",
        "official_mod_docs",
        "source_search",
    ]
    assert snapshot["next_capability"] == "registry_lookup"
