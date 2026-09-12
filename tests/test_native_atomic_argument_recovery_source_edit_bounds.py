from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.native_atomic_argument_recovery import (
    _page_result,
    _request,
    _source_edit_detail_schema,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def test_source_edit_detail_schema_preserves_unbounded_content_contract() -> None:
    detail = _source_edit_detail_schema(SOURCE_EDIT_SCHEMA, "create_file")

    assert detail["properties"]["path"]["maxLength"] == 256
    assert "maxLength" not in detail["properties"]["content"]

    content = "x" * 4096
    turn = SimpleNamespace(
        tool_calls=(
            SimpleNamespace(
                name="apply_source_edit",
                arguments={"path": "src/main/resources/mmm/large.txt", "content": content},
            ),
        )
    )

    arguments, error, _ = _page_result(
        turn,
        detail,
        SOURCE_EDIT_SCHEMA,
        "apply_source_edit",
    )

    assert error == ""
    assert arguments == {
        "path": "src/main/resources/mmm/large.txt",
        "content": content,
    }


def test_source_edit_request_allows_only_declared_long_text_exception() -> None:
    detail = _source_edit_detail_schema(SOURCE_EDIT_SCHEMA, "create_file")

    with pytest.raises(Exception, match="MODEL_ATOMICITY_STRING_UNBOUNDED"):
        assert_atomic_model_schema(detail, surface="generic model template")

    request = GenerationRequest(
        messages=({"role": "user", "content": "create the source file"},),
        tools=(),
        tool_choice="auto",
    )
    recovered = _request(
        request,
        page_index=2,
        page_count=2,
        page_schema=detail,
        action_name="apply_source_edit",
    )

    parameters = recovered.tools[0]["function"]["parameters"]
    assert parameters["properties"]["path"]["maxLength"] == 256
    assert "maxLength" not in parameters["properties"]["content"]
