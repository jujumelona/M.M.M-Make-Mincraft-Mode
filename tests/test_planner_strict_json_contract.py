from __future__ import annotations
import pytest
from minecraft_mod_ai import complete_planner as planner
from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai.planner_strict_json_contract import install
from minecraft_mod_ai.spec import SpecValidationError
EXPECTED = (frozenset({'value'}),)

def _extract(text: str, expected=EXPECTED):
    install(runtime)
    return runtime._extract_with_safe_empty_defaults(planner, text, expected_contracts=expected)

def test_exact_contract_json_is_accepted() -> None:
    assert _extract('{"value": 1}') == {'value': 1}

def test_transport_prose_around_one_complete_object_is_accepted() -> None:
    assert _extract('Here is the result:\n{"value": 1}\nDone.') == {'value': 1}

def test_response_wide_json_fence_is_accepted() -> None:
    assert _extract('```json\n{"value": 1}\n```') == {'value': 1}

def test_response_wide_bare_fence_is_accepted() -> None:
    assert _extract('```\n{"value": 1}\n```') == {'value': 1}

def test_arbitrary_fence_label_cannot_change_json_semantics() -> None:
    assert _extract('```text\n{"value": 1}\n```') == {'value': 1}

def test_qwen_think_transport_is_accepted_when_visible_json_is_unique() -> None:
    assert _extract('<think>internal transport reasoning</think>\n```json\n{"value": 1}\n```') == {'value': 1}

def test_preopened_qwen_think_close_is_transport_only() -> None:
    assert _extract('</think>\n{"value": 1}') == {'value': 1}

def test_utf8_bom_and_transport_text_are_irrelevant() -> None:
    assert _extract('\ufeffassistant output:\n{"value": 1}') == {'value': 1}

def test_multiple_outline_pages_can_leave_host_continuation_open() -> None:
    expected = (frozenset({'production_batches', 'complete', 'next_cursor'}),)
    payload = '{"production_batches":[1,2],"complete":true,"next_cursor":"p2"}\n{"production_batches":[3,4],"complete":false,"next_cursor":"continue_host"}'
    assert _extract(payload, expected) == {'production_batches': [1, 2, 3, 4], 'complete': False, 'next_cursor': 'continue_host'}

def test_intermediate_page_bookkeeping_does_not_discard_later_pages() -> None:
    expected = (frozenset({'production_batches', 'complete', 'next_cursor'}),)
    payload = '{"production_batches":[1],"complete":true,"next_cursor":""}\n{"production_batches":[2],"complete":true,"next_cursor":""}'
    assert _extract(payload, expected) == {'production_batches': [1, 2], 'complete': True, 'next_cursor': ''}

def test_outline_sequence_rejects_unrelated_json_object() -> None:
    expected = (frozenset({'production_batches', 'complete', 'next_cursor'}),)
    with pytest.raises(SpecValidationError, match='valid sequential production-outline'):
        _extract('{"production_batches":[],"complete":false,"next_cursor":"p2"}\n{"note":"alternative"}', expected)

def test_truncated_json_is_not_auto_closed() -> None:
    with pytest.raises(SpecValidationError, match='exactly one complete strict JSON object'):
        _extract('{"value": 1')

def test_two_complete_top_level_objects_are_rejected_for_nonpaginated_contract() -> None:
    with pytest.raises(SpecValidationError, match='exactly one complete strict JSON object'):
        _extract('{"value": 1}\n{"value": 2}')

def test_array_wrapping_the_object_is_rejected() -> None:
    with pytest.raises(SpecValidationError, match='top-level JSON value must be an object'):
        _extract('[{"value": 1}]')

def test_nested_objects_do_not_count_as_multiple_top_level_values() -> None:
    expected = (frozenset({'value', 'nested'}),)
    assert _extract('{"value": 1, "nested": {"child": true}}', expected) == {'value': 1, 'nested': {'child': True}}

def test_complete_json_inside_think_plus_visible_json_is_rejected_as_ambiguous() -> None:
    with pytest.raises(SpecValidationError, match='exactly one complete strict JSON object'):
        _extract('<think>{"value": 0}</think>\n{"value": 1}')

def test_extra_top_level_field_is_rejected() -> None:
    with pytest.raises(SpecValidationError, match='do not match the host contract'):
        _extract('{"value": 1, "unexpected": true}')
