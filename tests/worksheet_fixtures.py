"""Structured design fixtures shared by planning integration tests."""

from collections.abc import Mapping
from typing import Any

from minecraft_mod_ai.planning_detail_slots import (
    DETAIL_RECORDS,
    concern_record_schema,
)


def flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one canonical nested record to its unique leaf field names."""
    result: dict[str, Any] = {}

    def visit(value: Mapping[str, Any]) -> None:
        for key, item in value.items():
            if isinstance(item, Mapping):
                visit(item)
                continue
            if key in result:
                raise AssertionError(f"duplicate canonical leaf field: {key}")
            result[str(key)] = item

    visit(record)
    return result


def _canonical_fixture_record(section: str, concern: str) -> dict[str, Any]:
    schema = concern_record_schema(section, concern)

    def build(node: Mapping[str, Any]) -> dict[str, Any]:
        properties = node.get("properties")
        required = node.get("required")
        assert isinstance(properties, Mapping)
        assert isinstance(required, list)

        result: dict[str, Any] = {}
        for raw_field in required:
            field = str(raw_field)
            child = properties[field]
            assert isinstance(child, Mapping)
            if child.get("type") == "object":
                result[field] = build(child)
            else:
                result[field] = (
                    f"{section} {concern} {field}: server owns the observable outcome."
                )
        return result

    return build(schema)


def specification(section):
    if section == "state_model":
        return {
            "variables": [{
                "name": "credits",
                "owner": "server",
                "type": "Double",
                "unit": "credits",
                "default": "0",
                "domain": "economy",
            }],
            "transitions": [{
                "from_state": "idle",
                "trigger": "buy",
                "guard": "credits >= cost",
                "mutation": "credits -= cost",
                "to_state": "done",
            }],
            "invariants": [{
                "condition": "credits >= 0",
                "enforcement": "reject negative balance",
            }],
            "initialization": [{
                "owner": "server",
                "trigger": "server_start",
                "initial_state": "credits = 0",
            }],
            "updates": [{
                "trigger": "reward",
                "mutation": "credits += amount",
                "owner": "server",
            }],
            "cleanup": [{
                "event": "reset",
                "action": "credits = 0",
                "retained_state": "no state retained after reset",
            }],
            "concurrency": [{
                "entry_path": "state_runtime",
                "ownership": "server",
                "reentrancy_rule": "serialized",
            }],
            "inapplicable_concerns": [],
        }
    return {
        **{
            concern: [_canonical_fixture_record(section, concern)]
            for concern in DETAIL_RECORDS[section]
        },
        "inapplicable_concerns": [],
    }


def row(section, refs=()):
    return {
        "specification": specification(section),
        "constraint_evidence_refs": list(refs),
    }
