from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_planner as planner
from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai.planner_strict_json_contract import install
from minecraft_mod_ai.spec import SpecValidationError


EXPECTED = (frozenset({"value"}),)


def _extract(text: str, expected=EXPECTED):
    install(runtime)
    return runtime._extract_with_safe_empty_defaults(
        planner,
        text,
        expected_contracts=expected,
    )


def test_exact_contract_json_is_accepted() -> None:
    assert _extract('{"value": 1}') == {"value": 1}


def test_transport_prose_around_one_complete_object_is_accepted() -> None:
    assert _extract('Here is the result:\n{"value": 1}\nDone.') == {"value": 1}


def test_response_wide_json_fence_is_accepted() -> None:
    assert _extract('```json\n{"value": 1}\n```') == {"value": 1}


def test_response_wide_bare_fence_is_accepted() -> None:
    assert _extract('```\n{"value": 1}\n```') == {"value": 1}


def test_arbitrary_fence_label_cannot_change_json_semantics() -> None:
    assert _extract('```text\n{"value": 1}\n```') == {"value": 1}


def test_qwen_think_transport_is_accepted_when_visible_json_is_unique() -> None:
    assert _extract(
        '<think>internal transport reasoning</think>\n```json\n{"value": 1}\n```'
    ) == {"value": 1}


def test_preopened_qwen_think_close_is_transport_only() -> None:
    assert _extract('</think>\n{"value": 1}') == {"value": 1}


def test_utf8_bom_and_transport_text_are_irrelevant() -> None:
    assert _extract('\ufeffassistant output:\n{"value": 1}') == {"value": 1}


def test_production_outline_survives_unknown_transport_envelope() -> None:
    expected = (
        frozenset({"modules", "assets", "audio", "acceptance_tests"}),
        frozenset({"module_batches", "assets", "audio", "acceptance_tests"}),
        frozenset({"production_batches", "complete", "next_cursor"}),
    )
    payload = (
        '<|im_start|>assistant\n'
        '```json\n'
        '{"production_batches":[],"complete":true,"next_cursor":""}'
        '\n```\n<|im_end|>'
    )
    assert _extract(payload, expected) == {
        "production_batches": [],
        "complete": True,
        "next_cursor": "",
    }


def test_truncated_json_is_not_auto_closed() -> None:
    with pytest.raises(SpecValidationError, match="exactly one complete strict JSON object"):
        _extract('{"value": 1')


def test_two_complete_top_level_objects_are_rejected() -> None:
    with pytest.raises(SpecValidationError, match="found 2 complete outermost JSON containers"):
        _extract('{"value": 1}\n{"value": 2}')


def test_array_wrapping_the_object_is_rejected() -> None:
    with pytest.raises(SpecValidationError, match="top-level JSON value must be an object"):
        _extract('[{"value": 1}]')


def test_nested_objects_do_not_count_as_multiple_top_level_values() -> None:
    expected = (frozenset({"value", "nested"}),)
    assert _extract(
        '{"value": 1, "nested": {"child": true}}',
        expected,
    ) == {"value": 1, "nested": {"child": True}}


def test_complete_json_inside_think_plus_visible_json_is_rejected_as_ambiguous() -> None:
    with pytest.raises(SpecValidationError, match="found 2 complete outermost JSON containers"):
        _extract(
            '<think>{"value": 0}</think>\n{"value": 1}'
        )


def test_extra_top_level_field_is_rejected() -> None:
    with pytest.raises(SpecValidationError, match="do not match the host contract"):
        _extract('{"value": 1, "unexpected": true}')
