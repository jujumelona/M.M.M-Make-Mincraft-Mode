from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.spec import SpecValidationError
from minecraft_mod_ai.structured_subtree_repair_contract import _generate_section_exact


_SCHEMA = {
    "machine": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "power": {
                "type": "object",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["min", "max"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "power"],
        "additionalProperties": False,
    }
}


class _Router:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        if not self.outputs:
            raise AssertionError("unexpected planner generation")
        return self.outputs.pop(0)


def _run(router: _Router):
    return _generate_section_exact(
        router,
        prompt="Create one machine with a bounded power configuration.",
        section_id="systems",
        fields=("machine",),
        properties=_SCHEMA,
        research={},
        media_paths=(),
        trace_metadata=None,
    )


def test_nested_leaf_repair_preserves_valid_siblings():
    router = _Router(
        [
            json.dumps(
                {
                    "section": {
                        "machine": {
                            "name": "forge",
                            "power": {"min": 2, "max": "invalid"},
                        }
                    }
                }
            ),
            json.dumps({"repair": 8}),
        ]
    )

    result = _run(router)

    assert result == {
        "machine": {"name": "forge", "power": {"min": 2, "max": 8}}
    }
    assert len(router.calls) == 2
    repair_schema = router.calls[1]["response_schema"]
    assert repair_schema["properties"]["repair"] == {"type": "number"}
    payload = json.loads(router.calls[1]["messages"][1]["content"])
    assert payload["repair_path"] == "$.section.machine.power.max"
    assert payload["frozen_parent_context"]["min"] == 2


def test_unexpected_nested_field_is_deleted_without_model_repair_call():
    router = _Router(
        [
            json.dumps(
                {
                    "section": {
                        "machine": {
                            "name": "forge",
                            "power": {"min": 2, "max": 8, "debug": True},
                        }
                    }
                }
            )
        ]
    )

    result = _run(router)

    assert result["machine"]["power"] == {"min": 2, "max": 8}
    assert len(router.calls) == 1


def test_same_validator_predicate_cannot_oscillate_forever():
    router = _Router(
        [
            json.dumps(
                {
                    "section": {
                        "machine": {
                            "name": "forge",
                            "power": {"min": 2, "max": "bad-0"},
                        }
                    }
                }
            ),
            json.dumps({"repair": "bad-1"}),
            json.dumps({"repair": "bad-2"}),
            json.dumps({"repair": "bad-3"}),
        ]
    )

    with pytest.raises(SpecValidationError, match="made no validator progress"):
        _run(router)

    assert len(router.calls) == 4
