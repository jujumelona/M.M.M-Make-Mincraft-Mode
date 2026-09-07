from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.planning_contract_ssot import (
    RESEARCH_NOTE_SCHEMA,
    SUBMIT_PROMPT_STATE_SCHEMA,
    SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
    assert_all_planning_contracts_valid,
    is_schema_definition_echo,
    pre_design_support_schema,
    schema_skeleton_template,
)
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.planning_state_resolution import compile_researched_requirements
from minecraft_mod_ai.worksheet_atomic_chunker import (
    pack_section_concerns,
    worksheet_chunk_prompt,
    worksheet_chunk_schema,
)


def _assert_all_objects_closed(schema: Any, path: str = "$") -> None:
    if isinstance(schema, Mapping):
        if schema.get("type") == "object" or "properties" in schema:
            assert (
                schema.get("additionalProperties") is False
            ), f"Schema at {path} must set additionalProperties=False"
        for key, child in schema.items():
            _assert_all_objects_closed(child, f"{path}.{key}")
    elif isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        for idx, child in enumerate(schema):
            _assert_all_objects_closed(child, f"{path}[{idx}]")


def test_ssot_self_verification_passes() -> None:
    assert_all_planning_contracts_valid()


def test_all_planning_schemas_are_strictly_closed() -> None:
    schemas = [
        ("SUBMIT_PROMPT_STATE_SCHEMA", SUBMIT_PROMPT_STATE_SCHEMA),
        (
            "SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA",
            SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
        ),
        ("RESEARCH_NOTE_SCHEMA", RESEARCH_NOTE_SCHEMA),
        ("pre_design_support_schema", pre_design_support_schema(3)),
    ]
    for name, schema in schemas:
        _assert_all_objects_closed(schema, path=name)
        assert_atomic_model_schema(schema, surface=name)

    for section in DETAIL_RECORDS:
        chunks = pack_section_concerns(section)
        for index, concerns in enumerate(chunks, start=1):
            chunk_schema = worksheet_chunk_schema(
                section, concerns, include_evidence=(index == 1)
            )
            name = f"chunk:{section}:{index}"
            _assert_all_objects_closed(chunk_schema, path=name)
            assert_atomic_model_schema(chunk_schema, surface=name)


def test_schema_skeleton_template_contains_no_schema_keywords() -> None:
    for section in DETAIL_RECORDS:
        chunks = pack_section_concerns(section)
        for index, concerns in enumerate(chunks, start=1):
            chunk_schema = worksheet_chunk_schema(
                section, concerns, include_evidence=(index == 1)
            )
            skeleton = schema_skeleton_template(chunk_schema)
            text = json.dumps(skeleton)
            assert '"properties"' not in text
            assert '"additionalProperties"' not in text
            assert '"required"' not in text
            assert '"type": "object"' not in text
            assert '"type": "array"' not in text
            assert '"type": "string"' not in text


def test_is_schema_definition_echo() -> None:
    echoed = {
        "type": "object",
        "description": "chunk",
        "properties": {
            "preconditions": {"type": "array"},
        },
        "required": ["preconditions"],
        "additionalProperties": False,
    }
    assert is_schema_definition_echo(echoed) is True

    real_data = {
        "preconditions": [
            {"condition": "Player is crouching", "rejection": "Ignored"}
        ],
        "inputs": [],
        "inapplicable_concerns": [],
    }
    assert is_schema_definition_echo(real_data) is False


def test_worksheet_chunk_prompt_uses_skeleton_not_raw_schema() -> None:
    prompt = worksheet_chunk_prompt(
        "behavior_contract", 1, 3, ("actors", "entry_conditions"), include_evidence=True
    )
    assert '"additionalProperties"' not in prompt
    assert '"properties":' not in prompt
    assert "Fill and return only a JSON object matching this exact data template skeleton:" in prompt


class _BuggyConfigRouter:
    def generate_tool_decision(self, *args, **kwargs):
        raise ModelConfigurationError("Simulated configuration error in tool schema")


def test_compile_researched_requirements_fails_fast_on_config_error() -> None:
    from minecraft_mod_ai.planning_state_contract import _sha
    state = {
        "schema_version": "mmm/planning-state-v1",
        "original_prompt": "test prompt",
        "prompt_sha256": _sha("test prompt"),
        "state_sha256": "",
        "plan_ready": False,
        "goal": {"statement": "test goal", "source_quote": "test"},
        "known": [],
        "references": [],
        "scope_status": "unspecified",
        "unresolved": [],
        "research_queue": [],
        "evidence": [],
        "resolved": [],
        "decisions": [],
        "implementation_candidates": [],
        "coverage": [],
        "blockers": [],
    }
    from minecraft_mod_ai.planning_state_resolution import _rehash
    state = _rehash(state)

    router = _BuggyConfigRouter()
    with pytest.raises(ModelConfigurationError, match="Simulated configuration error"):
        compile_researched_requirements(router, "test prompt", state)
