from types import SimpleNamespace

import pytest

from minecraft_mod_ai.generation_output_budget import generation_output_token_budget
from minecraft_mod_ai.planner_hole_filling import PlanningHoleFillError, _fill_page
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


@pytest.mark.parametrize(
    "repair", [_block(1, "Repair B only"), '{"bad": "unterminated']
)
def test_repair_keeps_accepted_neighbor_and_never_accepts_malformed_json(repair):
    class Router:
        calls = 0

        def generate_text(self, role, messages, **kwargs):
            self.calls += 1
            assert kwargs == {"response_format": "text", "enable_tools": False}
            if self.calls == 1:
                return _block(1, "Keep A") + "### Hole 2\nDecision: truncated"
            import json

            packet = json.loads(messages[-1]["content"].split("\n", 1)[1])
            assert packet["modules"][0]["implementation_template"]["holes"] == [
                {"hole_id": "b"}
            ]
            assert current_output_limit() == 384
            return repair

    trace = _Trace()
    module = {
        "module_id": "host",
        "implementation_template": {
            "holes": [
                {"hole_id": "a"},
                {"hole_id": "b"},
            ]
        },
    }
    if repair.startswith("{"):
        with pytest.raises(PlanningHoleFillError, match="bounded repair"):
            _fill_page(Router(), module, trace)
        assert trace.attempts[-1]["raw_output"] == repair
    else:
        fills = _fill_page(Router(), module, trace)
        assert [fill["implementation_decision"] for fill in fills] == [
            "Keep A",
            "Repair B only",
        ]
    assert len(trace.attempts) == 2


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
