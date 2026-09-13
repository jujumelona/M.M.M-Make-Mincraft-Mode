from pathlib import Path
import subprocess


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


# Restore only the semantic-frontier pagination function from the verified runtime
# commit. Keep later unrelated changes in this module intact.
path = "minecraft_mod_ai/planning_state_resolution.py"
current = read(path)
prior = subprocess.check_output(
    ["git", "show", "6d5d1c6fa97dd395ddadad3d4382791655ad5397:" + path],
    text=True,
)
start_marker = "def _generate_requirement_pages("
end_marker = "def compile_researched_requirements("
cur_start = current.index(start_marker)
cur_end = current.index(end_marker, cur_start)
old_start = prior.index(start_marker)
old_end = prior.index(end_marker, old_start)
current = current[:cur_start] + prior[old_start:old_end] + current[cur_end:]
current = current.replace(
    "from .planning_contract_ssot import (\n    REQUIREMENT_COVERAGE_SCHEMA,\n    SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,\n)\n",
    "from .planning_contract_ssot import SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA\n",
    1,
)
write(path, current)


# The retired coverage-review schema has no runtime owner after semantic-frontier
# pagination is restored, so remove it from the SSOT instead of leaving dead contract code.
path = "minecraft_mod_ai/planning_contract_ssot.py"
text = read(path)
start = text.index("REQUIREMENT_COVERAGE_SCHEMA: dict[str, Any] = {")
end = text.index("RESEARCH_NOTE_SCHEMA: dict[str, Any] = {", start)
text = text[:start] + text[end:]
entry = '        ("REQUIREMENT_COVERAGE_SCHEMA", REQUIREMENT_COVERAGE_SCHEMA),\n'
if entry not in text:
    raise SystemExit("coverage schema registry entry missing")
text = text.replace(entry, "", 1)
write(path, text)


# Tests describe the host-owned pagination protocol, not a second planner verdict call.
write(
    "tests/test_planning_requirement_pagination.py",
    '''import json

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
    assert continuation["original_prompt"] == PROMPT
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
''',
)


# Same-stage generators mutate shared stage state and must stay serialized.
path = "tests/test_scheduler_parallel_safety_contract.py"
text = read(path)
old = '''def test_anchor_fenced_runtime_disables_coarse_stage_admission() -> None:
    assert scheduler_contract._SERIAL_CPU_STAGES == ()
'''
new = '''def test_shared_stage_mutation_domains_are_serialized_by_admission() -> None:
    assert scheduler_contract._SERIAL_CPU_STAGES == (
        "generate:content",
        "generate:system",
        "generate:entity",
    )
'''
if old not in text:
    raise SystemExit("scheduler stale-contract anchor missing")
write(path, text.replace(old, new, 1))
