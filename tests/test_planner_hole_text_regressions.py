from types import SimpleNamespace

import pytest

from minecraft_mod_ai.generation_output_budget import generation_output_token_budget
from minecraft_mod_ai.planner_hole_filling import _fill_page
from minecraft_mod_ai.planner_hole_text import parse_hole_text
from minecraft_mod_ai.planner_operation import current_output_limit, planner_operation


def _block(
    index, decision='Use a "quoted" symbol and C:\\state without JSON escaping.'
):
    return (
        f"### Hole {index}\nDecision: {decision}\nSteps:\n- Validate state.\n"
        "- Apply the approved transition.\nVerification: Run the supplied gate.\n"
    )


class _Trace:
    def __init__(self):
        self.attempts = []

    def record_attempt(self, **kwargs):
        self.attempts.append(kwargs)


def test_text_protocol_preserves_quotes_and_discards_truncated_block():
    fills = parse_hole_text(
        _block(1) + "### Hole 2\nDecision: unfinished",
        [
            {"hole_id": "host_a"},
            {"hole_id": "host_b"},
        ],
    )
    assert [fill["hole_id"] for fill in fills] == ["host_a"]
    assert '"quoted"' in fills[0]["implementation_decision"]
    assert "C:\\state" in fills[0]["implementation_decision"]
    with pytest.raises(ValueError, match="repeated"):
        parse_hole_text(_block(1) + _block(1), [{"hole_id": "host_a"}])


def test_partial_refinement_keeps_neighbor_host_default_without_retry():
    class Router:
        calls = 0

        def generate_text(self, role, messages, **kwargs):
            self.calls += 1
            assert kwargs == {"response_format": "text", "enable_tools": False}
            assert current_output_limit() == 640
            return _block(1, "Refine A") + "### Hole 2\nDecision: truncated"

    router = Router()
    trace = _Trace()
    module = {
        "module_id": "host",
        "implementation_template": {
            "holes": [
                {"hole_id": "a", "kind": "implementation", "subject": "A"},
                {"hole_id": "b", "kind": "implementation", "subject": "B"},
            ]
        },
    }

    fills = _fill_page(router, module, trace)

    assert router.calls == 1
    assert len(trace.attempts) == 1
    assert fills[0]["implementation_decision"] == "Refine A"
    assert fills[0]["fill_source"] == "model_refined_host_template"
    assert fills[1]["hole_id"] == "b"
    assert fills[1]["fill_source"] == "host_template_default"
    assert fills[1]["implementation_decision"]
    assert fills[1]["local_steps"]
    assert fills[1]["verification_intent"]


def test_malformed_refinement_is_ignored_and_host_defaults_complete_plan():
    class Router:
        calls = 0

        def generate_text(self, role, messages, **kwargs):
            self.calls += 1
            return '{"bad": "unterminated'

    router = Router()
    trace = _Trace()
    module = {
        "module_id": "host",
        "implementation_template": {
            "holes": [
                {"hole_id": "a", "kind": "implementation", "subject": "A"},
                {"hole_id": "b", "kind": "implementation", "subject": "B"},
            ]
        },
    }

    fills = _fill_page(router, module, trace)

    assert router.calls == 1
    assert len(trace.attempts) == 1
    assert {fill["hole_id"] for fill in fills} == {"a", "b"}
    assert all(fill["fill_source"] == "host_template_default" for fill in fills)
    assert all(fill["implementation_decision"] for fill in fills)
    assert all(fill["verification_intent"] for fill in fills)


def test_operation_budget_caps_dynamic_context_and_restores_after_failure():
    config = SimpleNamespace(
        adapter="llama_cpp",
        max_new_tokens=27000,
        context_window=32768,
        extra={"dynamic_output_budget": True},
    )
    assert generation_output_token_budget(config) > 768
    with pytest.raises(RuntimeError):
        with planner_operation("research.facet", output_tokens=768):
            assert generation_output_token_budget(config) == 768
            with planner_operation("nested", output_tokens=1024):
                assert current_output_limit() == 768
            raise RuntimeError("failure")
    assert current_output_limit() is None
