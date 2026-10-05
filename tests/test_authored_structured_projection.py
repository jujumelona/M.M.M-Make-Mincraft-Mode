from __future__ import annotations

from minecraft_mod_ai.authored_structured_design import (
    _projection_record,
    _projection_text,
)


def test_projection_preserves_nested_typed_ir_without_leaf_collisions() -> None:
    guard = {
        "kind": "and",
        "terms": [
            {"kind": "state_ref", "name": "resource_economy_mode"},
            {
                "kind": "not",
                "term": {
                    "kind": "state_ref",
                    "name": "spacecraft_build_mode",
                },
            },
        ],
    }
    mutation = [
        {
            "target": "resource_economy_mode",
            "operator": "=",
            "value": {"kind": "literal", "value": True},
        }
    ]
    record = {
        "from_state": "idle",
        "trigger": "tick",
        "guard": guard,
        "mutation": mutation,
        "to_state": "active",
    }

    projected = _projection_record(
        record,
        fields=("from_state", "trigger", "guard", "mutation", "to_state"),
    )

    assert projected["guard"] == guard
    assert projected["mutation"] == mutation
    rendered = _projection_text(projected["guard"])
    assert '"kind":"and"' in rendered
    assert rendered.count('"kind"') == 4


def test_projection_text_preserves_false_and_zero_scalars() -> None:
    assert _projection_text(False) == "False"
    assert _projection_text(0) == "0"
