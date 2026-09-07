from __future__ import annotations

from minecraft_mod_ai import planning_state_implementation as implementation


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


def _block(section: str, text: str) -> str:
    return f"<<<SECTION:{section}>>>\n{text}\n<<<END_SECTION>>>"


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


def test_requirement_details_use_one_model_call_when_batch_is_complete():
    router = _Router([
        "\n".join(
            (
                _block(
                    "behavior_contract",
                    "The server owns the exchange decision, validates eligibility, and emits one observable success or rejection result.",
                ),
                _block(
                    "state_model",
                    "The economy state has one authoritative owner, bounded numeric values, explicit transitions, and deterministic cleanup rules.",
                ),
            )
        )
    ])

    result = implementation._compile_requirement_specifications(
        router,
        requirement=REQUIREMENT,
        selected_sections=SECTIONS,
        evidence=EVIDENCE,
    )

    assert list(result) == list(SECTIONS)
    assert len(router.calls) == 1


def test_requirement_details_repair_only_missing_batch_section():
    router = _Router([
        _block(
            "behavior_contract",
            "The server owns the exchange decision, validates eligibility, and emits one observable success or rejection result.",
        ),
        "The state owner validates every mutation, keeps values bounded, and resets transient state on lifecycle cleanup.",
    ])

    result = implementation._compile_requirement_specifications(
        router,
        requirement=REQUIREMENT,
        selected_sections=SECTIONS,
        evidence=EVIDENCE,
    )

    assert result["behavior_contract"].startswith("The server owns")
    assert result["state_model"].startswith("The state owner")
    assert len(router.calls) == 2


def test_requirement_details_fall_back_to_single_sections_when_batch_fails():
    router = _Router([
        RuntimeError("batch transport failure"),
        "The server validates the interaction inputs, applies one bounded mutation, and exposes success or rejection to the player.",
        "The authoritative state owner stores bounded values, applies guarded transitions, and performs deterministic lifecycle cleanup.",
    ])

    result = implementation._compile_requirement_specifications(
        router,
        requirement=REQUIREMENT,
        selected_sections=SECTIONS,
        evidence=EVIDENCE,
    )

    assert list(result) == list(SECTIONS)
    assert len(router.calls) == 3
