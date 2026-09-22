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
    assert "apply_source_edit" not in names
    assert {"search_code_rag", "inspect_modrinth_project", "external_mcp_call"} <= names


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


def test_api_namespace_compile_errors_require_recovery_evidence() -> None:
    assert loop._diagnostics_require_api_evidence((
        {
            "message": (
                "package net.minecraft.registry does not exist\n"
                "import net.minecraft.registry.Registry;"
            )
        },
    ))
    assert not loop._diagnostics_require_api_evidence((
        {"message": "cannot find symbol\nsymbol: variable localCounter"},
    ))


def test_rejected_alternate_evidence_query_maps_to_single_forced_route() -> None:
    search_code = {
        "type": "function",
        "function": {
            "name": "search_code_rag",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "index_path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }
    recovered = loop._forced_evidence_recovery_arguments(
        (search_code,),
        "search_code_rag",
        (
            {
                "failure_code": "TOOL_NOT_VISIBLE",
                "original_tool": "search_project_rag",
                "raw_arguments": (
                    '{"query":"AuthoredFeature011 current implementation",'
                    '"minecraft_version":26.2,"limit":8}'
                ),
            },
        ),
    )
    assert recovered == {
        "query": "AuthoredFeature011 current implementation",
        "limit": 8,
    }


def test_compile_backed_api_failure_routes_recover_before_edit() -> None:
    source = inspect.getsource(loop._generate_with_tools_impl)
    marker = 'current_compile_errors = tuple(state.latest_verifier_errors)'
    assert marker in source
    branch = source.split(marker, 1)[1].split(
        'state.validation_status = "DEFERRED"', 1
    )[0]
    assert "_diagnostics_require_api_evidence" in branch
    assert "LoopPhase.RECOVER" in branch
