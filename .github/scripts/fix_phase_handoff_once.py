from __future__ import annotations

from pathlib import Path


LOOP_PATH = Path("minecraft_mod_ai/progress_aware_tool_loop.py")
TEST_PATH = Path("tests/test_tool_call_admission_contract.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one {label}; found {text.count(old)}")
    return text.replace(old, new, 1)


text = LOOP_PATH.read_text(encoding="utf-8")
if "def _compact_phase_tool_transcript(" in text:
    raise SystemExit("phase transcript helper already exists")

marker = "\ndef _generate_with_tools_impl(\n"
helper = '''

def _compact_phase_tool_transcript(
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
text = replace_once(text, marker, helper + marker, "_generate_with_tools_impl marker")

old_initial = '''    else:
        state.phase = LoopPhase.OBSERVE

    emit_root_cause(
'''
new_initial = '''    else:
        state.phase = LoopPhase.OBSERVE

    last_prompt_phase = state.phase

    emit_root_cause(
'''
text = replace_once(text, old_initial, new_initial, "initial phase block")

old_loop = '''    while True:
        if (
            implementation_requires_mutation
            and state.workspace_changed
'''
new_loop = '''    while True:
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
            implementation_requires_mutation
            and state.workspace_changed
'''
text = replace_once(text, old_loop, new_loop, "main tool loop entry")
LOOP_PATH.write_text(text, encoding="utf-8")


tests = TEST_PATH.read_text(encoding="utf-8")
if "test_phase_handoff_closes_old_tool_protocol_and_preserves_observation_data" in tests:
    raise SystemExit("phase handoff tests already exist")

tests += r'''


def test_phase_handoff_closes_old_tool_protocol_and_preserves_observation_data():
    from minecraft_mod_ai.progress_aware_tool_loop import (
        LoopPhase,
        _compact_phase_tool_transcript,
    )

    messages = [
        {"role": "user", "content": "implement the target"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
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
            "tool_call_id": "call_search",
            "name": "search_code_rag",
            "content": "public final class DebugToken {}",
        },
    ]

    removed = _compact_phase_tool_transcript(
        messages,
        previous_phase=LoopPhase.OBSERVE,
        next_phase=LoopPhase.ACT,
    )

    assert removed == 2
    assert all(message.get("role") != "tool" for message in messages)
    assert all(not message.get("tool_calls") for message in messages)
    assert messages[-1]["role"] == "system"
    assert "OBSERVE->ACT" in messages[-1]["content"]
    assert "public final class DebugToken {}" in messages[-1]["content"]
    assert "search_code_rag" not in messages[-1]["content"]


def test_phase_handoff_is_wired_before_next_model_turn():
    import inspect

    from minecraft_mod_ai.progress_aware_tool_loop import _generate_with_tools_impl

    source = inspect.getsource(_generate_with_tools_impl)
    assert "state.phase != last_prompt_phase" in source
    assert "_compact_phase_tool_transcript(" in source
    assert "phase_tool_transcript_handoff" in source
'''
TEST_PATH.write_text(tests, encoding="utf-8")
