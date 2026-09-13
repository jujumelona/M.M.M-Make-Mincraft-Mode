import json

import pytest

from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.planning_state_contract import _build_host_state
from minecraft_mod_ai.planning_state_resolution import compile_researched_requirements

PROMPT = "Gather resources, earn currency, trade, build spacecraft parts, upgrade weapons and hire crew."
BEHAVIORS = [
    "Players gather resources.", "Players earn currency.", "Players trade resources.",
    "Players build spacecraft parts.", "Players upgrade weapons.", "Players hire crew.",
]


def state(prompt=PROMPT):
    return _build_host_state(prompt, {
        "goal": {"statement": prompt, "source_quote": prompt},
        "known": [{"statement": prompt, "source_quote": prompt}],
        "references": [], "scope_status": "explicit", "unresolved": [],
    })


def page(behaviors):
    return {"requirements": [
        {"statement": text, "semantic_capability": text, "acceptance": text}
        for text in behaviors
    ]}


class Router:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.messages.append(json.loads(messages[1]["content"]))
        return next(self.responses)


def coverage(complete=False):
    return {"complete": complete,
            "remaining_source_quote": "" if complete else "upgrade weapons and hire crew",
            "remaining_behavior": "" if complete else "Upgrade weapons and hire crew."}


def test_more_than_four_behaviors_survive_requirement_compilation():
    router = Router([page(BEHAVIORS[:4]), coverage(), page(BEHAVIORS[4:])])
    result = compile_researched_requirements(router, PROMPT, state())
    requirements = [row for row in result["decisions"] if row["decision_type"] == "requirement"]
    assert [row["statement"] for row in requirements] == BEHAVIORS
    assert len(result["research_queue"]) == 6
    assert len(result["unresolved"]) == 6
    assert router.messages[1]["already_compiled_requirements"] == page(BEHAVIORS[:4])["requirements"]
    assert router.messages[2]["uncovered_authored_behavior"] == coverage()


def test_full_final_page_stops_on_coverage_without_forcing_another_generation():
    router = Router([page(BEHAVIORS[:4]), coverage(True)])
    result = compile_researched_requirements(router, PROMPT, state())
    assert len(result["decisions"]) == 4
    assert len(router.messages) == 2


def test_repeated_full_page_cannot_be_mistaken_for_complete_coverage():
    router = Router([page(BEHAVIORS[:4]), coverage(), page(BEHAVIORS[:4])])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_NO_PROGRESS"):
        compile_researched_requirements(router, PROMPT, state())


def test_continuation_failure_cannot_publish_only_first_four_requirements():
    router = Router([page(BEHAVIORS[:4]), coverage()])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_FAILED"):
        compile_researched_requirements(router, PROMPT, state())


def test_missing_coverage_decision_cannot_certify_full_page():
    router = Router([page(BEHAVIORS[:4]), {}])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_COVERAGE_INVALID"):
        compile_researched_requirements(router, PROMPT, state())


def test_invented_remaining_quote_cannot_trigger_more_generation():
    router = Router([page(BEHAVIORS[:4]), {**coverage(), "remaining_source_quote": "unrequested teleporter"}])
    with pytest.raises(ModelConfigurationError, match="exact task quote"):
        compile_researched_requirements(router, PROMPT, state())
    assert len(router.messages) == 2


def test_coverage_schema_is_atomic():
    from minecraft_mod_ai.model_output_atomicity_contract import (
        assert_atomic_model_schema,
    )
    from minecraft_mod_ai.planning_contract_ssot import REQUIREMENT_COVERAGE_SCHEMA
    assert_atomic_model_schema(REQUIREMENT_COVERAGE_SCHEMA, surface="requirement coverage")


def test_coverage_transport_failure_cannot_publish_partial_requirements():
    router = Router([page(BEHAVIORS[:4])])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_COVERAGE_FAILED"):
        compile_researched_requirements(router, PROMPT, state())


def test_two_full_pages_do_not_force_a_third_page_of_reworded_features():
    behaviors = BEHAVIORS + ["Players fight aliens.", "Players colonize planets."]
    prompt = PROMPT + " Fight aliens and colonize planets."
    router = Router([page(behaviors[:4]), coverage(), page(behaviors[4:]), coverage(True)])
    result = compile_researched_requirements(router, prompt, state(prompt))
    assert len(result["decisions"]) == 8
    assert len(router.messages) == 4
