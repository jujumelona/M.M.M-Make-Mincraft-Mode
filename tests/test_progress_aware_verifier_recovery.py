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
