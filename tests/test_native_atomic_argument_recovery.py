from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.native_atomic_argument_recovery import (
    _argument_pages,
    _canonical_source_edit_operation,
    _page_result,
    _page_schema,
    _source_edit_detail_schema,
    host_selected_argument_turn,
)


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


def _turn(arguments: dict[str, object], name: str = "submit_action") -> SimpleNamespace:
    return SimpleNamespace(
        tool_calls=(SimpleNamespace(name=name, arguments=arguments),)
    )


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


def test_generic_argument_decomposition_keeps_existing_bounded_pages() -> None:
    pages = _argument_pages(_parameters())
    assert [tuple(page["properties"]) for page in pages] == [
        ("actor", "preconditions", "inputs"),
        ("outputs", "constraint_evidence_refs"),
    ]


def test_source_edit_alias_is_canonicalized_before_detail_recovery() -> None:
    assert _canonical_source_edit_operation("create") == "create_file"
    assert _canonical_source_edit_operation("replace") == "replace_exact"
    assert _canonical_source_edit_operation("delete") == "delete_file"
    assert _canonical_source_edit_operation("insert_after") == "insert_after"


def test_create_file_detail_schema_excludes_other_operation_fields() -> None:
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA

    detail = _source_edit_detail_schema(SOURCE_EDIT_SCHEMA, "create_file")
    assert tuple(detail["properties"]) == ("path", "content")


def test_replace_exact_detail_schema_excludes_other_operation_fields() -> None:
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA

    detail = _source_edit_detail_schema(SOURCE_EDIT_SCHEMA, "replace_exact")
    assert tuple(detail["properties"]) == ("path", "old", "new", "count")


def test_discriminated_source_edit_recovery_drops_union_pollution() -> None:
    from minecraft_mod_ai.model_adapters.base import GenerationRequest, GenerationResponse
    from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA

    tool = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "test source mutation",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create DebugToken.java"},),
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
    )
    schemas_seen: list[tuple[str, ...]] = []

    def current(adapter, page_request):
        del adapter
        schema = page_request.response_schema or page_request.tools[0]["function"]["parameters"]
        properties = tuple(schema["properties"])
        schemas_seen.append(properties)
        if properties == ("operation",):
            # Reproduce a noisy model response: union members from unrelated operations
            # may still appear in arguments, but the selector owns operation only.
            return GenerationResponse(
                tool_calls=(
                    SimpleNamespace(
                        name="apply_source_edit",
                        arguments={
                            "operation": "create",
                            "old": "stale",
                            "anchor": "stale",
                            "member": "stale",
                        },
                    ),
                )
            )
        assert properties == ("path", "content")
        return GenerationResponse(
            tool_calls=(
                SimpleNamespace(
                    name="apply_source_edit",
                    arguments={
                        "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                        "content": "package dev.mmm.debugfixture;\nfinal class DebugToken {}\n",
                        "old": "pollution",
                        "new": "pollution",
                        "anchor": "pollution",
                        "count": 7,
                        "declaration": "pollution",
                        "import_name": "pollution",
                        "member": "pollution",
                        "package_name": "pollution",
                    },
                ),
            )
        )

    turn = host_selected_argument_turn(
        current,
        object(),
        request,
        "apply_source_edit",
        prefix="test",
    )

    assert schemas_seen == [("operation",), ("path", "content")]
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].arguments == {
        "operation": "create_file",
        "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
        "content": "package dev.mmm.debugfixture;\nfinal class DebugToken {}\n",
    }


def test_backend_boundary_reaches_canonical_owner_without_json_retry():
    import pytest
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )
    from minecraft_mod_ai.native_atomic_argument_recovery import _page_attempt

    boundary = LlamaCompletionBoundaryError(
        "prefill calibration unavailable",
        kind=OUTPUT_EXHAUSTED,
        partial_message={"content": '{"operation":"cre'},
        max_tokens=1,
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
