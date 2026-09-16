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
if "def _sync_phase_tool_transcript(" in text:
    raise SystemExit("phase sync helper already exists")

marker = "\ndef _generate_with_tools_impl(\n"
sync_helper = '''

def _sync_phase_tool_transcript(
    messages: list[dict[str, Any]],
    *,
    state: HostRunState,
    last_prompt_phase: LoopPhase,
    stage: str,
) -> LoopPhase:
    """Apply one phase-boundary handoff and return the phase now represented in messages."""
    next_phase = state.phase
    if next_phase == last_prompt_phase:
        return last_prompt_phase

    removed_protocol_messages = _compact_phase_tool_transcript(
        messages,
        previous_phase=last_prompt_phase,
        next_phase=next_phase,
    )
    emit_root_cause(
        "phase_tool_transcript_handoff",
        stage=stage,
        operation="generate_with_tools",
        gate="phase_boundary",
        result="PASS",
        reason="closed prior phase tool protocol before the next model turn",
        details={
            "previous_phase": last_prompt_phase.value,
            "next_phase": next_phase.value,
            "removed_protocol_messages": removed_protocol_messages,
            "message_count": len(messages),
        },
    )
    return next_phase
'''
text = replace_once(text, marker, sync_helper + marker, "generate implementation marker")

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
            messages,
            state=state,
            last_prompt_phase=last_prompt_phase,
            stage=stage,
        )

        if (
'''
text = replace_once(text, old_loop, new_loop, "inline phase handoff branch")
LOOP_PATH.write_text(text, encoding="utf-8")


tests = TEST_PATH.read_text(encoding="utf-8")
old_test = '''def test_phase_handoff_is_wired_before_next_model_turn():
    import inspect

    from minecraft_mod_ai.progress_aware_tool_loop import _generate_with_tools_impl

    source = inspect.getsource(_generate_with_tools_impl)
    assert "state.phase != last_prompt_phase" in source
    assert "_compact_phase_tool_transcript(" in source
    assert "phase_tool_transcript_handoff" in source
'''
new_test = '''def test_phase_handoff_sync_owns_transition_outside_main_loop():
    import inspect

    from minecraft_mod_ai.progress_aware_tool_loop import (
        HostRunState,
        LoopPhase,
        _generate_with_tools_impl,
        _sync_phase_tool_transcript,
    )

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search_code_rag", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "name": "search_code_rag",
            "content": "localized source body",
        },
    ]
    state = HostRunState(phase=LoopPhase.ACT)

    represented_phase = _sync_phase_tool_transcript(
        messages,
        state=state,
        last_prompt_phase=LoopPhase.OBSERVE,
        stage="generation",
    )

    assert represented_phase == LoopPhase.ACT
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "OBSERVE->ACT" in messages[0]["content"]
    assert "localized source body" in messages[0]["content"]

    main_source = inspect.getsource(_generate_with_tools_impl)
    sync_source = inspect.getsource(_sync_phase_tool_transcript)
    assert "last_prompt_phase = _sync_phase_tool_transcript(" in main_source
    assert "state.phase != last_prompt_phase" not in main_source
    assert "phase_tool_transcript_handoff" in sync_source
'''
tests = replace_once(tests, old_test, new_test, "phase handoff wiring test")
TEST_PATH.write_text(tests, encoding="utf-8")
