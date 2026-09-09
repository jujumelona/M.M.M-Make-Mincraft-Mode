from __future__ import annotations

from minecraft_mod_ai.planning_criterion_fragments import (
    generate_criterion_fragment,
    generate_criterion_fragments_batch,
)


_REQUIREMENT = {"statement": "Implement the requested behavior."}
_SECTION = "behavior_contract"


def _fragment(implementation: str = "Apply the requested behavior.") -> dict:
    return {
        "section_updates": [
            {
                "section": _SECTION,
                "implementation": implementation,
                "constraint": "",
                "evidence_refs": [],
            }
        ]
    }


class _Router:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = iter(responses)
        self.tool_calls: list[dict] = []

    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("criterion planning must not use raw structured text generation")

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description="",
    ):
        self.tool_calls.append(
            {
                "role": role,
                "messages": tuple(messages),
                "tool_name": tool_name,
                "parameters": parameters,
                "description": description,
            }
        )
        return next(self._responses)


def test_single_criterion_uses_one_forced_template_call():
    router = _Router([_fragment()])

    result = generate_criterion_fragment(
        router,
        requirement=_REQUIREMENT,
        criterion="The requested behavior is observable.",
        selected_sections=(_SECTION,),
        evidence=[],
        allowed_refs=set(),
    )

    assert result == _fragment()
    assert len(router.tool_calls) == 1
    call = router.tool_calls[0]
    assert call["role"] == "planner"
    assert call["tool_name"] == "submit_criterion_fragment"
    assert call["description"]
    assert call["parameters"]["additionalProperties"] is False
    prompt = "\n".join(str(message.get("content", "")) for message in call["messages"])
    assert "fixed template" in prompt.lower()
    assert "serialization syntax" in prompt.lower()
    assert "Return JSON only" not in prompt


def test_no_progress_repair_is_another_forced_template_fill():
    router = _Router([_fragment(""), _fragment("Apply the corrected behavior.")])

    result = generate_criterion_fragment(
        router,
        requirement=_REQUIREMENT,
        criterion="The requested behavior is observable.",
        selected_sections=(_SECTION,),
        evidence=[],
        allowed_refs=set(),
    )

    assert result == _fragment("Apply the corrected behavior.")
    assert [call["tool_name"] for call in router.tool_calls] == [
        "submit_criterion_fragment",
        "submit_criterion_fragment",
    ]
    repair_prompt = "\n".join(
        str(message.get("content", "")) for message in router.tool_calls[1]["messages"]
    )
    assert "Correction required" in repair_prompt


def test_batch_criteria_use_one_forced_batch_template_call():
    batch = {
        "criterion_fragments": [
            {"criterion_index": 0, **_fragment("Implement criterion zero.")},
            {"criterion_index": 1, **_fragment("Implement criterion one.")},
        ]
    }
    router = _Router([batch])

    result = generate_criterion_fragments_batch(
        router,
        requirement=_REQUIREMENT,
        criteria={0: "Criterion zero.", 1: "Criterion one."},
        selected_sections=(_SECTION,),
        evidence=[],
        allowed_refs=set(),
    )

    assert result == {
        0: _fragment("Implement criterion zero."),
        1: _fragment("Implement criterion one."),
    }
    assert len(router.tool_calls) == 1
    call = router.tool_calls[0]
    assert call["tool_name"] == "submit_criterion_fragments"
    assert call["parameters"]["additionalProperties"] is False
    prompt = "\n".join(str(message.get("content", "")) for message in call["messages"])
    assert "fixed template" in prompt.lower()
    assert "serialization syntax" in prompt.lower()
    assert "one JSON response" not in prompt
