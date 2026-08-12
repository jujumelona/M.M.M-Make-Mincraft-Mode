from __future__ import annotations

import pytest

from minecraft_mod_ai import complete_planner as planner
from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai.planner_strict_json_contract import install
from minecraft_mod_ai.spec import SpecValidationError


EXPECTED = (frozenset({"value"}),)


def test_embedded_json_with_prose_is_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            'Here is the result: {"value": 1}',
            expected_contracts=EXPECTED,
        )


def test_truncated_json_is_not_auto_closed() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            '{"value": 1',
            expected_contracts=EXPECTED,
        )


def test_exact_contract_json_is_accepted() -> None:
    install(runtime)
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        '{"value": 1}',
        expected_contracts=EXPECTED,
    ) == {"value": 1}


def test_response_wide_json_fence_is_accepted_as_transport_only() -> None:
    install(runtime)
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        '```json\n{"value": 1}\n```',
        expected_contracts=EXPECTED,
    ) == {"value": 1}


def test_leading_qwen_think_channel_is_transport_only() -> None:
    install(runtime)
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        '<think>internal reasoning not returned to the host</think>\n{"value": 1}',
        expected_contracts=EXPECTED,
    ) == {"value": 1}


def test_preopened_qwen_think_channel_close_is_transport_only() -> None:
    install(runtime)
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        '</think>\n{"value": 1}',
        expected_contracts=EXPECTED,
    ) == {"value": 1}


def test_qwen_think_channel_then_json_fence_is_accepted() -> None:
    install(runtime)
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        '<think>\n\n</think>\n```json\n{"value": 1}\n```',
        expected_contracts=EXPECTED,
    ) == {"value": 1}


def test_utf8_bom_is_transport_only() -> None:
    install(runtime)
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        '\ufeff{"value": 1}',
        expected_contracts=EXPECTED,
    ) == {"value": 1}


def test_production_outline_qwen_wrapper_and_fence_are_accepted() -> None:
    install(runtime)
    expected = (
        frozenset({"modules", "assets", "audio", "acceptance_tests"}),
        frozenset({"module_batches", "assets", "audio", "acceptance_tests"}),
        frozenset({"production_batches", "complete", "next_cursor"}),
    )
    payload = (
        '<think>\n\n</think>\n```json\n'
        '{"production_batches":[],"complete":true,"next_cursor":""}'
        '\n```'
    )
    assert runtime._extract_with_safe_empty_defaults(
        planner,
        payload,
        expected_contracts=expected,
    ) == {
        "production_batches": [],
        "complete": True,
        "next_cursor": "",
    }


def test_json_fence_with_surrounding_prose_is_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            'result follows\n```json\n{"value": 1}\n```',
            expected_contracts=EXPECTED,
        )


def test_prose_after_think_channel_is_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            '<think>reasoning</think>\nresult follows\n{"value": 1}',
            expected_contracts=EXPECTED,
        )


def test_unterminated_think_channel_is_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            '<think>reasoning\n{"value": 1}',
            expected_contracts=EXPECTED,
        )


def test_non_json_fence_is_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            '```text\n{"value": 1}\n```',
            expected_contracts=EXPECTED,
        )


def test_multiple_json_values_are_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="one complete strict JSON object"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            '{"value": 1}\n{"value": 2}',
            expected_contracts=EXPECTED,
        )


def test_extra_top_level_field_is_rejected() -> None:
    install(runtime)
    with pytest.raises(SpecValidationError, match="do not match the host contract"):
        runtime._extract_with_safe_empty_defaults(
            planner,
            '{"value": 1, "unexpected": true}',
            expected_contracts=EXPECTED,
        )
