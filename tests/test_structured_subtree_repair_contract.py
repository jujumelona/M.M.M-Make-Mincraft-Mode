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


def test_nested_leaf_repair_preserves_valid_siblings_and_constrains_leaf_schema():
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
    assert router.calls[0]["response_format"] == "text"
    assert router.calls[0]["response_schema"] is None
    assert router.calls[1]["response_format"] == "json"
    repair_schema = router.calls[1]["response_schema"]
    assert repair_schema["properties"]["repair"] == {"type": "number"}
    payload = json.loads(router.calls[1]["messages"][1]["content"])
    assert payload["repair_path"] == "$.section.machine.power.max"
    assert payload["frozen_parent_context"]["min"] == 2


def test_progression_type_failure_is_repaired_as_array_without_regenerating_siblings():
    properties = {
        "progression": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "combat": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
        "mod_context": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    router = _Router(
        [
            json.dumps(
                {
                    "section": {
                        "progression": "level up",
                        "combat": {"boss": ["server authoritative"]},
                        "mod_context": {"scope": ["maple progression"]},
                    }
                }
            ),
            json.dumps({"repair": ["level up", "enhance equipment"]}),
        ]
    )

    result = _generate_section_exact(
        router,
        prompt="메이플 스타일 성장 시스템을 설계해줘",
        section_id="systems_and_progression",
        fields=("progression", "combat", "mod_context"),
        properties=properties,
        research={},
        media_paths=(),
        trace_metadata=None,
    )

    assert result["progression"] == ["level up", "enhance equipment"]
    assert result["combat"] == {"boss": ["server authoritative"]}
    assert result["mod_context"] == {"scope": ["maple progression"]}
    assert len(router.calls) == 2
    assert router.calls[1]["response_schema"]["properties"]["repair"]["type"] == "array"


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


def test_same_validator_predicate_fails_after_one_nonprogressing_leaf_repair():
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
            json.dumps({"repair": "still-not-a-number"}),
        ]
    )

    with pytest.raises(SpecValidationError, match="made no validator progress"):
        _run(router)

    assert len(router.calls) == 2
