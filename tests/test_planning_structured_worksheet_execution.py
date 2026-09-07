from __future__ import annotations

import json

import minecraft_mod_ai.planning_state_implementation as planning
from minecraft_mod_ai.planning_detail_template import normalize_required_sections


class _WorksheetRouter:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(self.payload)


def _worksheet_payload(sections: tuple[str, ...]) -> dict[str, object]:
    return {
        section: {
            "specification": (
                f"{section} owns a distinct concrete contract with explicit inputs, "
                f"branches, failure behavior, and observable outcome for this requirement."
            ),
            "constraint_evidence_refs": [],
        }
        for section in sections
    }


def test_detailed_planning_uses_one_schema_constrained_worksheet_call() -> None:
    sections = normalize_required_sections()
    router = _WorksheetRouter(_worksheet_payload(sections))
    requirement = {
        "requirement_id": "req_001",
        "statement": "Implement one observable gameplay requirement.",
        "acceptance": ["The expected gameplay result is observable."],
    }
    evidence = [
        {
            "research_ref": "research_001",
            "claims": ["A grounded implementation constraint exists."],
            "evidence_refs": ["evidence_001"],
            "source": "fixture",
            "sufficient": True,
        }
    ]

    worksheet = planning._compile_requirement_worksheet(
        router,
        requirement=requirement,
        selected_sections=sections,
        evidence=evidence,
        allowed={"evidence_001"},
    )

    assert tuple(worksheet) == sections
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["role"] == "planner"
    assert call["response_format"] == "json"
    assert call["enable_tools"] is False
    schema = call["response_schema"]
    assert schema["required"] == list(sections)
    assert schema["additionalProperties"] is False


def test_free_form_detailed_section_path_is_not_present() -> None:
    assert not hasattr(planning, "_plain_section")
    assert not hasattr(planning, "_strip_leading_meta_reasoning")
    assert not hasattr(planning, "_normalize_section_text")
