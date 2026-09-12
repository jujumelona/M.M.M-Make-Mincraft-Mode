from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.native_atomic_argument_recovery import (
    _page_result,
    _source_edit_detail_schema,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def test_source_edit_detail_schema_preserves_unbounded_content_contract() -> None:
    detail = _source_edit_detail_schema(SOURCE_EDIT_SCHEMA, "create_file")

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
