from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.native_atomic_argument_recovery import _page_result, _page_schema


def _parameters() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "actor": {"type": "string"},
            "preconditions": {"type": "array", "items": {"type": "string"}},
            "inputs": {"type": "array", "items": {"type": "string"}},
            "outputs": {"type": "array", "items": {"type": "string"}},
            "constraint_evidence_refs": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "actor",
            "preconditions",
            "inputs",
            "outputs",
            "constraint_evidence_refs",
        ],
        "additionalProperties": False,
    }


def _turn(arguments: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(content=json.dumps(arguments))


def test_page_result_ignores_full_schema_field_owned_by_later_page() -> None:
    parameters = _parameters()
    page = _page_schema(
        parameters,
        ("actor", "preconditions", "inputs", "outputs"),
    )

    arguments, error, _ = _page_result(
        _turn(
            {
                "actor": "player",
                "preconditions": [],
                "inputs": [],
                "outputs": [],
                "constraint_evidence_refs": ["constraint-1"],
            }
        ),
        page,
        parameters,
    )

    assert error == ""
    assert arguments == {
        "actor": "player",
        "preconditions": [],
        "inputs": [],
        "outputs": [],
    }


def test_page_result_keeps_unknown_field_visible_to_strict_validator() -> None:
    parameters = _parameters()
    page = _page_schema(
        parameters,
        ("actor", "preconditions", "inputs", "outputs"),
    )

    arguments, error, _ = _page_result(
        _turn(
            {
                "actor": "player",
                "preconditions": [],
                "inputs": [],
                "outputs": [],
                "invented_field": "must fail",
            }
        ),
        page,
        parameters,
    )

    assert arguments is None
    assert "failed the host page schema" in error
    assert "invented_field" in error


def test_page_result_still_rejects_missing_required_page_field() -> None:
    parameters = _parameters()
    page = _page_schema(
        parameters,
        ("actor", "preconditions", "inputs", "outputs"),
    )

    arguments, error, _ = _page_result(
        _turn({"constraint_evidence_refs": ["constraint-1"]}),
        page,
        parameters,
    )

    assert arguments is None
    assert "failed the host page schema" in error


def test_later_page_owns_its_field_even_if_earlier_field_is_repeated() -> None:
    parameters = _parameters()
    page = _page_schema(parameters, ("constraint_evidence_refs",))

    arguments, error, _ = _page_result(
        _turn(
            {
                "actor": "player",
                "constraint_evidence_refs": ["constraint-1"],
            }
        ),
        page,
        parameters,
    )

    assert error == ""
    assert arguments == {"constraint_evidence_refs": ["constraint-1"]}


def test_backend_boundary_reaches_canonical_owner_without_json_retry():
    import pytest
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED, LlamaCompletionBoundaryError,
    )
    from minecraft_mod_ai.native_atomic_argument_recovery import _page_attempt
    boundary = LlamaCompletionBoundaryError(
        "prefill calibration unavailable", kind=OUTPUT_EXHAUSTED,
        partial_message={"content": '{"operation":"cre'}, max_tokens=1,
    )
    calls = []

    def current(adapter, request):
        calls.append(request)
        raise boundary

    with pytest.raises(LlamaCompletionBoundaryError) as caught:
        _page_attempt(current, None, object(), {}, {})
    assert caught.value is boundary
    assert caught.value.partial_message["content"] == '{"operation":"cre'
    assert len(calls) == 1


def test_cancellation_is_not_rewritten_as_invalid_arguments():
    import pytest
    from minecraft_mod_ai.native_atomic_argument_recovery import _page_attempt

    def cancel(adapter, request):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _page_attempt(cancel, None, object(), {}, {})
