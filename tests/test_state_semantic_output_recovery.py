"""State planner fallback must stay bounded and semantically verified."""

from __future__ import annotations

import pytest

from minecraft_mod_ai import planner_state_authoring as state
from minecraft_mod_ai.llama_finish_reason_contract import (
    LlamaCompletionBoundaryError,
    OUTPUT_EXHAUSTED,
)


def _exhausted() -> LlamaCompletionBoundaryError:
    return LlamaCompletionBoundaryError(
        "output exhausted", kind=OUTPUT_EXHAUSTED,
        prompt_tokens=739, completion_tokens=814, max_tokens=814,
    )


def test_single_state_field_recovers_peg_failure_without_dropping_semantic_gate(
    monkeypatch,
) -> None:
    schemas = []
    costs = []

    def generate(_router, _role, _messages, **kwargs):
        schemas.append(kwargs["response_schema"])
        if len(schemas) == 1:
            raise _exhausted()
        return {"label": "planet_ore"}

    monkeypatch.setattr(state, "generate_fixed_template_value", generate)
    output = state.author_state_semantic_page(
        None, "Add a planet ore type",
        concern="resources", fields=("label",), count=1,
        item_schema={"properties": {
            "label": {"type": "string", "maxLength": 32, "pattern": "^[a-z_]+$"},
        }},
        on_additional_call=lambda: costs.append(1),
    )
    assert output == {"resources": [{"label": "planet_ore"}]}
    assert "pattern" in schemas[0]["properties"]["label"]
    assert "pattern" not in schemas[1]["properties"]["label"]
    assert costs == [1]


def test_scalar_recovery_does_not_accept_invalid_meaning(monkeypatch) -> None:
    calls = []

    def generate(_router, _role, _messages, **kwargs):
        calls.append(kwargs["response_schema"])
        if len(calls) == 1:
            raise _exhausted()
        return {"label": "{invalid}"}

    monkeypatch.setattr(state, "generate_fixed_template_value", generate)
    with pytest.raises(ValueError, match="STATE_SEMANTIC_FIELD_INVALID"):
        state.author_state_semantic_page(
            None, "Add a planet ore type",
            concern="resources", fields=("label",), count=1,
            item_schema={"properties": {
                "label": {"type": "string", "maxLength": 32, "pattern": "^[a-z_]+$"},
            }},
        )
    assert len(calls) == 2


def test_multi_field_output_exhaustion_splits_only_failed_row(monkeypatch) -> None:
    calls = []
    additional = []

    def generate(_router, _role, _messages, **kwargs):
        fields = tuple(kwargs["response_schema"]["required"])
        calls.append(fields)
        if fields == ("type", "label"):
            raise _exhausted()
        return {"type": "number"} if fields == ("type",) else {"label": "ship_power"}

    monkeypatch.setattr(state, "generate_fixed_template_value", generate)
    output = state.author_state_semantic_page(
        None, "Add ship power",
        concern="resources", fields=("type", "label"), count=1,
        item_schema={"properties": {
            "type": {"type": "string", "enum": ["number", "boolean"]},
            "label": {"type": "string", "maxLength": 32},
        }},
        on_additional_call=lambda: additional.append(1),
    )
    assert output == {"resources": [{"type": "number", "label": "ship_power"}]}
    assert calls == [("type", "label"), ("type",), ("label",)]
    assert additional == [1, 1]


def test_unsplittable_second_exhaustion_is_not_silenced(monkeypatch) -> None:
    calls = []

    def generate(_router, _role, _messages, **kwargs):
        calls.append(kwargs["response_schema"])
        raise _exhausted()

    monkeypatch.setattr(state, "generate_fixed_template_value", generate)
    with pytest.raises(LlamaCompletionBoundaryError):
        state.author_state_semantic_page(
            None, "Build a spaceship",
            concern="resources", fields=("label",), count=1,
            item_schema={"properties": {
                "label": {"type": "string", "maxLength": 32},
            }},
        )
    assert len(calls) == 2


def test_scalar_fallback_consumes_budget_before_second_call(monkeypatch) -> None:
    calls = []

    def generate(_router, _role, _messages, **kwargs):
        calls.append(kwargs["response_schema"])
        raise _exhausted()

    def budget_denied():
        raise RuntimeError("budget exceeded")

    monkeypatch.setattr(state, "generate_fixed_template_value", generate)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        state.author_state_semantic_page(
            None, "Build a spaceship",
            concern="resources", fields=("label",), count=1,
            item_schema={"properties": {
                "label": {"type": "string", "maxLength": 32},
            }},
            on_additional_call=budget_denied,
        )
    assert len(calls) == 1
