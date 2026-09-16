from __future__ import annotations

from pathlib import Path


LOOP_PATH = Path("minecraft_mod_ai/progress_aware_tool_loop.py")
TEST_PATH = Path("tests/test_tool_call_admission_contract.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


text = LOOP_PATH.read_text(encoding="utf-8")
old_owner = '''def _compact_phase_tool_transcript(
    messages: list[dict[str, Any]],
    *,
    previous_phase: LoopPhase,
    next_phase: LoopPhase,
) -> int:
    """Close the previous phase tool protocol while preserving observation data."""
    if previous_phase == next_phase:
        return 0

    compacted: list[dict[str, Any]] = []
    observations: list[str] = []
    removed = 0

    for raw_message in messages:
        message = dict(raw_message)
        role = str(message.get("role") or "")
        if role == "assistant" and message.get("tool_calls"):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                compacted.append({"role": "assistant", "content": content})
            removed += 1
            continue
        if role == "tool":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                observations.append(content)
            removed += 1
            continue
        compacted.append(message)

    if not removed:
        return 0

    handoff_lines = [
        f"MMM_PHASE_HANDOFF {previous_phase.value}->{next_phase.value}",
        "The previous phase is complete. Prior tool-call protocol is closed and must not be repeated.",
        "The observation data below is non-executable context. In the new phase, only the current request tool schema and tool_choice are callable.",
    ]
    for index, observation in enumerate(observations, start=1):
        handoff_lines.append(f"Observation {index}:\\n{observation}")
    compacted.append({"role": "system", "content": "\\n".join(handoff_lines)})
    messages[:] = compacted
    return removed
'''
new_owner = '''def _sync_phase_tool_transcript(
    messages: list[dict[str, Any]],
    *,
    state: HostRunState,
    last_prompt_phase: LoopPhase,
    stage: str,
) -> LoopPhase:
    """Close prior-phase tool protocol and preserve only its observation data."""
    next_phase = state.phase
    if next_phase == last_prompt_phase:
        return last_prompt_phase
    compacted: list[dict[str, Any]] = []
    observations: list[str] = []
    removed = 0
    for raw_message in messages:
        message = dict(raw_message)
        role = str(message.get("role") or "")
        content = message.get("content")
        if role == "tool":
            if isinstance(content, str) and content.strip():
                observations.append(content)
            removed += 1
            continue
        if role == "assistant" and message.get("tool_calls"):
            if isinstance(content, str) and content.strip():
                compacted.append({"role": "assistant", "content": content})
            removed += 1
            continue
        compacted.append(message)
    if removed:
        handoff = [
            f"MMM_PHASE_HANDOFF {last_prompt_phase.value}->{next_phase.value}",
            "Prior phase tool-call protocol is closed; do not repeat it.",
            "Observations below are non-executable context. Only current tool schemas are callable.",
            *(f"Observation {index}:\\n{value}" for index, value in enumerate(observations, 1)),
        ]
        compacted.append({"role": "system", "content": "\\n".join(handoff)})
        messages[:] = compacted
    emit_root_cause(
        "phase_tool_transcript_handoff",
        stage=stage,
        operation="generate_with_tools",
        gate="phase_boundary",
        result="PASS",
        reason="closed prior phase tool protocol before the next model turn",
        details={"previous_phase": last_prompt_phase.value, "next_phase": next_phase.value,
                 "removed_protocol_messages": removed, "message_count": len(messages)},
    )
    return next_phase
'''
text = replace_once(text, old_owner, new_owner, "phase transcript owner")

old_loop = '''    while True:
        if state.phase != last_prompt_phase:
            previous_prompt_phase = last_prompt_phase
            removed_protocol_messages = _compact_phase_tool_transcript(
                messages,
                previous_phase=previous_prompt_phase,
                next_phase=state.phase,
            )
            emit_root_cause(
                "phase_tool_transcript_handoff",
                stage=stage,
                operation="generate_with_tools",
                gate="phase_boundary",
                result="PASS",
                reason="closed prior phase tool protocol before the next model turn",
                details={
                    "previous_phase": previous_prompt_phase.value,
                    "next_phase": state.phase.value,
                    "removed_protocol_messages": removed_protocol_messages,
                    "message_count": len(messages),
                },
            )
            last_prompt_phase = state.phase

        if (
'''
new_loop = '''    while True:
        last_prompt_phase = _sync_phase_tool_transcript(
            messages, state=state, last_prompt_phase=last_prompt_phase, stage=stage
        )

        if (
'''
text = replace_once(text, old_loop, new_loop, "inline phase handoff branch")
LOOP_PATH.write_text(text, encoding="utf-8")


tests = TEST_PATH.read_text(encoding="utf-8")
start = tests.index("def test_phase_handoff_closes_old_tool_protocol_and_preserves_observation_data():")
replacement = '''def test_phase_handoff_closes_old_protocol_and_preserves_observation_data():
    from minecraft_mod_ai.progress_aware_tool_loop import HostRunState, LoopPhase, _sync_phase_tool_transcript

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "search", "type": "function", "function": {"name": "search_code_rag", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "search", "name": "search_code_rag", "content": "public final class DebugToken {}"},
    ]
    phase = _sync_phase_tool_transcript(
        messages, state=HostRunState(phase=LoopPhase.ACT),
        last_prompt_phase=LoopPhase.OBSERVE, stage="generation",
    )
    assert phase == LoopPhase.ACT
    assert len(messages) == 1 and messages[0]["role"] == "system"
    assert "OBSERVE->ACT" in messages[0]["content"]
    assert "public final class DebugToken {}" in messages[0]["content"]
    assert "search_code_rag" not in messages[0]["content"]


def test_phase_handoff_is_generic_and_main_loop_has_no_transition_branch():
    import inspect
    from minecraft_mod_ai.progress_aware_tool_loop import HostRunState, LoopPhase, _generate_with_tools_impl, _sync_phase_tool_transcript

    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "edit", "type": "function", "function": {"name": "apply_source_edit", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "edit", "name": "apply_source_edit", "content": "workspace_changed=true; sha256=abc123"},
    ]
    phase = _sync_phase_tool_transcript(
        messages, state=HostRunState(phase=LoopPhase.VERIFY),
        last_prompt_phase=LoopPhase.ACT, stage="generation",
    )
    assert phase == LoopPhase.VERIFY
    assert "ACT->VERIFY" in messages[0]["content"]
    assert "workspace_changed=true" in messages[0]["content"]
    assert "apply_source_edit" not in messages[0]["content"]
    source = inspect.getsource(_generate_with_tools_impl)
    assert "last_prompt_phase = _sync_phase_tool_transcript(" in source
    assert "state.phase != last_prompt_phase" not in source
'''
tests = tests[:start] + replacement
TEST_PATH.write_text(tests, encoding="utf-8")
