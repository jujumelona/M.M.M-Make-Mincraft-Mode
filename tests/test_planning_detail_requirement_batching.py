from __future__ import annotations

import pytest

from minecraft_mod_ai import planning_state_implementation as implementation
from minecraft_mod_ai.planner_operation import current_output_limit


SECTIONS = ("behavior_contract", "state_model")
REQUIREMENT = {
    "requirement_id": "req_001",
    "statement": "A player can exchange collected resources through a bounded economy loop.",
}
EVIDENCE = [
    {
        "research_ref": "research_001",
        "claims": ["The implementation boundary is grounded for this test."],
        "evidence_refs": ["evidence_001"],
        "sufficient": True,
        "source": "fixture",
    }
]


class _Router:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.output_limits = []

    def generate_text(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.output_limits.append(current_output_limit())
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_requirement_details_generate_each_selected_semantic_section_once():
    router = _Router(
        [
            "The server owns the exchange decision, validates eligibility, and emits one observable success or rejection result.",
            "The economy state has one authoritative owner, bounded numeric values, explicit transitions, and deterministic cleanup rules.",
        ]
    )

    result = implementation._compile_requirement_specifications(
        router,
        requirement=REQUIREMENT,
        selected_sections=SECTIONS,
        evidence=EVIDENCE,
    )

    assert list(result) == list(SECTIONS)
    assert len(router.calls) == len(SECTIONS)
    assert router.output_limits == [None, None]


def test_later_semantic_section_receives_completed_section_as_continuity_context():
    first = (
        "The server owns the exchange decision, validates eligibility, and emits one observable success or rejection result."
    )
    router = _Router(
        [
            first,
            "The economy state has one authoritative owner, bounded numeric values, explicit transitions, and deterministic cleanup rules.",
        ]
    )

    implementation._compile_requirement_specifications(
        router,
        requirement=REQUIREMENT,
        selected_sections=SECTIONS,
        evidence=EVIDENCE,
    )

    second_messages = router.calls[1][0][1]
    second_prompt = second_messages[1]["content"]
    assert "Earlier completed semantic sections" in second_prompt
    assert f"- behavior_contract: {first}" in second_prompt
    assert "<<<SECTION:" not in second_prompt


def test_semantic_section_transport_failure_is_not_retried_or_fallback_rewritten():
    router = _Router([RuntimeError("transport failure")])

    with pytest.raises(RuntimeError, match="transport failure"):
        implementation._compile_requirement_specifications(
            router,
            requirement=REQUIREMENT,
            selected_sections=SECTIONS,
            evidence=EVIDENCE,
        )

    assert len(router.calls) == 1


def test_section_normalizer_strips_complete_leading_think_block():
    result = implementation._normalize_section_text(
        "<think>I should reason about ownership before answering.</think>\n"
        "The server owns the exchange mutation and emits a deterministic observable result.",
        "behavior_contract",
    )

    assert result == (
        "The server owns the exchange mutation and emits a deterministic observable result."
    )


def test_section_normalizer_keeps_only_explicit_final_after_reasoning_label():
    result = implementation._normalize_section_text(
        "Thinking Process: I should first reason about every possible state transition.\n\n"
        "Specification: The authoritative state owner validates each transition and bounds every stored value.",
        "state_model",
    )

    assert result == (
        "The authoritative state owner validates each transition and bounds every stored value."
    )


def test_section_normalizer_rejects_reasoning_label_without_final_boundary():
    with pytest.raises(ValueError, match="DETAILED_PLAN_META_REASONING"):
        implementation._normalize_section_text(
            "Analysis: I should inspect all possible branches before deciding how to implement this section.",
            "behavior_contract",
        )


def test_section_normalizer_preserves_legitimate_analysis_word_in_prose():
    text = (
        "The analysis state belongs to the server and is cleared deterministically when the lifecycle ends."
    )

    assert implementation._normalize_section_text(text, "state_model") == text
