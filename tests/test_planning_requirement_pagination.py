import json

import pytest

from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.planning_state_contract import _build_host_state
from minecraft_mod_ai.planning_state_resolution import compile_researched_requirements

PROMPT = "Gather resources, earn currency, trade, build spacecraft parts, upgrade weapons and hire crew."
BEHAVIORS = [
    "Players gather resources.",
    "Players earn currency.",
    "Players trade resources.",
    "Players build spacecraft parts.",
    "Players upgrade weapons.",
    "Players hire crew.",
]


def state(prompt=PROMPT):
    return _build_host_state(
        prompt,
        {
            "goal": {"statement": prompt, "source_quote": prompt},
            "known": [{"statement": prompt, "source_quote": prompt}],
            "references": [],
            "scope_status": "explicit",
            "unresolved": [],
        },
    )


def page(behaviors):
    return {
        "requirements": [
            {"statement": text, "semantic_capability": text, "acceptance": text}
            for text in behaviors
        ]
    }


class Router:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def generate_tool_decision(self, role, messages, **kwargs):
        self.messages.append(json.loads(messages[1]["content"]))
        return next(self.responses)


def test_more_than_four_behaviors_survive_requirement_compilation():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[4:])])
    result = compile_researched_requirements(router, PROMPT, state())
    requirements = [
        row for row in result["decisions"] if row["decision_type"] == "requirement"
    ]
    assert [row["statement"] for row in requirements] == BEHAVIORS
    assert result["research_queue"] == []
    assert result["unresolved"] == []
    assert (
        router.messages[1]["already_compiled_requirements"]
        == page(BEHAVIORS[:4])["requirements"]
    )
    assert "uncovered_authored_behavior" not in router.messages[1]


def test_full_page_stops_when_next_semantic_frontier_is_empty():
    router = Router([page(BEHAVIORS[:4]), page([])])
    result = compile_researched_requirements(router, PROMPT, state())
    assert len(result["decisions"]) == 4
    assert len(router.messages) == 2


def test_repeated_full_page_is_semantic_convergence_evidence():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[:4])])
    result = compile_researched_requirements(router, PROMPT, state())
    assert len(result["decisions"]) == 4
    assert len(router.messages) == 2


def test_continuation_failure_cannot_publish_partial_requirements():
    router = Router([page(BEHAVIORS[:4])])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_FAILED"):
        compile_researched_requirements(router, PROMPT, state())


def test_invalid_continuation_payload_cannot_certify_completion():
    router = Router([page(BEHAVIORS[:4]), {}])
    with pytest.raises(
        ModelConfigurationError,
        match="REQUIREMENT_PAGINATION_FAILED: invalid continuation page",
    ):
        compile_researched_requirements(router, PROMPT, state())


def test_continuation_context_is_host_owned_not_model_coverage_authority():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[4:])])
    compile_researched_requirements(router, PROMPT, state())
    continuation = router.messages[1]
    assert continuation["task"]["original_prompt"] == PROMPT
    assert (
        continuation["already_compiled_requirements"]
        == page(BEHAVIORS[:4])["requirements"]
    )
    assert "complete" not in continuation
    assert "remaining_source_quote" not in continuation
    assert "remaining_behavior" not in continuation
    assert "uncovered_authored_behavior" not in continuation


def test_requirement_page_schema_is_atomic():
    from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
    from minecraft_mod_ai.planning_contract_ssot import (
        SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
    )

    assert_atomic_model_schema(
        SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
        surface="requirement page",
    )


def test_continuation_transport_failure_cannot_publish_partial_requirements():
    class FailingRouter(Router):
        def generate_tool_decision(self, role, messages, **kwargs):
            self.messages.append(json.loads(messages[1]["content"]))
            if len(self.messages) > 1:
                raise ConnectionError("planner transport lost")
            return page(BEHAVIORS[:4])

    router = FailingRouter([])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_FAILED"):
        compile_researched_requirements(router, PROMPT, state())


def test_two_full_pages_stop_on_empty_frontier_without_reworded_features():
    behaviors = BEHAVIORS + ["Players fight aliens.", "Players colonize planets."]
    prompt = PROMPT + " Fight aliens and colonize planets."
    router = Router([page(behaviors[:4]), page(behaviors[4:]), page([])])
    result = compile_researched_requirements(router, prompt, state(prompt))
    assert len(result["decisions"]) == 8
    assert len(router.messages) == 3
