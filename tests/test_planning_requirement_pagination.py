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


def state():
    return _build_host_state(PROMPT, {
        "goal": {"statement": PROMPT, "source_quote": PROMPT},
        "known": [{"statement": PROMPT, "source_quote": PROMPT}],
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


def test_more_than_four_behaviors_survive_requirement_compilation():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[4:])])
    result = compile_researched_requirements(router, PROMPT, state())
    requirements = [row for row in result["decisions"] if row["decision_type"] == "requirement"]
    assert [row["statement"] for row in requirements] == BEHAVIORS
    assert len(result["research_queue"]) == 6
    assert len(result["unresolved"]) == 6
    assert router.messages[1]["already_compiled_requirements"] == page(BEHAVIORS[:4])["requirements"]


def test_full_final_page_requires_empty_continuation_without_fallback_requirement():
    router = Router([page(BEHAVIORS[:4]), page([])])
    result = compile_researched_requirements(router, PROMPT, state())
    assert len(result["decisions"]) == 4
    assert len(router.messages) == 2


def test_repeated_full_page_cannot_be_mistaken_for_complete_coverage():
    router = Router([page(BEHAVIORS[:4]), page(BEHAVIORS[:4])])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_NO_PROGRESS"):
        compile_researched_requirements(router, PROMPT, state())


def test_continuation_failure_cannot_publish_only_first_four_requirements():
    router = Router([page(BEHAVIORS[:4])])
    with pytest.raises(ModelConfigurationError, match="REQUIREMENT_PAGINATION_FAILED"):
        compile_researched_requirements(router, PROMPT, state())
