from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import planning_state_implementation as implementation
from minecraft_mod_ai.planning_detail_template import CORE_WORKSHEET_SECTIONS


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


def _payload() -> dict[str, dict[str, object]]:
    return {
        section: {
            "specification": (
                f"{section} defines one concrete authoritative contract with bounded failure "
                "behavior and an observable verification outcome."
            ),
            "constraint_evidence_refs": [],
        }
        for section in CORE_WORKSHEET_SECTIONS
    }


class _Router:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_text(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_requirement_details_generate_selected_sections_in_one_schema_call() -> None:
    router = _Router([json.dumps(_payload())])

    result = implementation._compile_requirement_worksheet(
        router,
        requirement=REQUIREMENT,
        selected_sections=CORE_WORKSHEET_SECTIONS,
        evidence=EVIDENCE,
        allowed={"evidence_001"},
    )

    assert tuple(result) == CORE_WORKSHEET_SECTIONS
    assert len(router.calls) == 1
    kwargs = router.calls[0][1]
    assert kwargs["response_format"] == "json"
    assert kwargs["response_schema"]["required"] == list(CORE_WORKSHEET_SECTIONS)
    assert kwargs["enable_tools"] is False


def test_structured_worksheet_transport_failure_is_not_retried() -> None:
    router = _Router([RuntimeError("transport failure")])

    with pytest.raises(RuntimeError, match="transport failure"):
        implementation._compile_requirement_worksheet(
            router,
            requirement=REQUIREMENT,
            selected_sections=CORE_WORKSHEET_SECTIONS,
            evidence=EVIDENCE,
            allowed={"evidence_001"},
        )

    assert len(router.calls) == 1


def test_old_free_form_section_parser_does_not_exist() -> None:
    assert not hasattr(implementation, "_compile_requirement_specifications")
    assert not hasattr(implementation, "_normalize_section_text")
    assert not hasattr(implementation, "_continuity_context")
