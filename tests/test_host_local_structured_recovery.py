from __future__ import annotations

import json

from minecraft_mod_ai import structured_repair_contract as field_local
from minecraft_mod_ai import structured_subtree_repair_contract as exact


class Router:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def test_exact_repair_keeps_invalid_candidate_available_for_leaf_repair():
    router = Router(
        [
            json.dumps({"section": {"x": {"a": "keep", "b": "bad"}}}),
            json.dumps({"repair": 4}),
        ]
    )
    result = exact._generate_section_exact(
        router,
        prompt="x",
        section_id="x",
        fields=("x",),
        properties={
            "x": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            }
        },
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert result == {"x": {"a": "keep", "b": 4}}
    assert len(router.calls) == 2
    assert router.calls[0]["response_format"] == "text"
    assert router.calls[0]["response_schema"] is None
    assert router.calls[1]["response_format"] == "json"
    assert router.calls[1]["response_schema"]["properties"]["repair"] == {
        "type": "number"
    }


def test_dispatcher_path_keeps_raw_initial_boundary_but_constrains_repair():
    router = Router(
        [
            json.dumps({"section": {"name": 3}}),
            json.dumps({"repair": "ok"}),
        ]
    )
    result = field_local._generate_section_local(
        router,
        prompt="x",
        section_id="x",
        fields=("name",),
        properties={"name": {"type": "string", "minLength": 1}},
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert result == {"name": "ok"}
    assert len(router.calls) == 2
    assert router.calls[0]["response_format"] == "text"
    assert router.calls[0]["response_schema"] is None
    assert router.calls[1]["response_format"] == "json"
    assert router.calls[1]["response_schema"]["properties"]["repair"] == {
        "type": "string",
        "minLength": 1,
    }
