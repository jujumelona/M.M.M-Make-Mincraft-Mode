from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, ValidationError

from minecraft_mod_ai import planner_json_runtime_contract as runtime
from minecraft_mod_ai import production_stream_efficiency_contract as stream


def _production_page(**overrides):
    page = {
        "modules": [],
        "assets": [],
        "audio": [],
        "acceptance_tests": [],
        "completed_deliverables": [],
        "complete": False,
        "next_cursor": "host_remaining_1",
    }
    page.update(overrides)
    return page


def _schema():
    return runtime._schema_for_contract(_production_page())


def test_production_schema_rejects_all_empty_concrete_outputs() -> None:
    validator = Draft202012Validator(_schema())
    with pytest.raises(ValidationError):
        validator.validate(_production_page())


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "modules",
            [
                {
                    "module_id": "platform_lock",
                    "kind": "config",
                    "config": {},
                    "depends_on": [],
                    "required_gates": [],
                }
            ],
        ),
        ("assets", [{}]),
        ("audio", [{}]),
        ("acceptance_tests", ["platform lock is host-verifiable"]),
    ],
)
def test_production_schema_accepts_each_concrete_output_family(field, value) -> None:
    validator = Draft202012Validator(_schema())
    validator.validate(_production_page(**{field: value}))


def test_stream_retry_budget_cannot_undercut_canonical_production_budget() -> None:
    assert stream._FULL_PAGE_DECODE_LIMIT >= runtime._attempt_budget(True)
