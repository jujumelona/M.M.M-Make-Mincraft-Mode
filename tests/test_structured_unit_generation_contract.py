from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import agentic_research_game_design as design
from minecraft_mod_ai.spec import SpecValidationError
from minecraft_mod_ai.structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)


def _systems_schema() -> dict:
    section_id, fields, properties = design._SECTION_SPECS[1]
    assert section_id == "systems_and_progression"
    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": dict(properties),
                "required": list(fields),
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }


def test_unrelated_structured_transport_still_validates_json_shape() -> None:
    raw = json.dumps(
        {
            "section": {
                "progression": ["gather", "launch"],
                "combat": ["alien_combat", "colony_defense"],
                "mod_context": {},
            }
        }
    )

    validated = validate_structured_output(
        raw,
        response_format="json",
        response_schema=_systems_schema(),
    )

    assert json.loads(validated)["section"]["combat"] == [
        "alien_combat",
        "colony_defense",
    ]


def test_design_section_owner_rejects_missing_required_markdown_heading_once() -> None:
    class Router:
        def __init__(self) -> None:
            self.calls = 0
            self.kwargs = None

        def generate_text(self, *_args, **kwargs) -> str:
            self.calls += 1
            self.kwargs = kwargs
            return """## progression
- gather
- launch
## mod_context
### persistence
- save progression
"""

    router = Router()
    _section_id, fields, _properties = design._SECTION_SPECS[1]
    with pytest.raises(SpecValidationError, match="combat"):
        design._generate_section(
            router,
            prompt="우주 모드를 설계해줘",
            section_id="systems_and_progression",
            fields=fields,
            research={},
            media_paths=(),
            trace_metadata=None,
        )

    assert router.calls == 1
    assert router.kwargs["response_format"] == "text"
    assert router.kwargs["response_schema"] is None


def test_modules_have_one_canonical_five_column_markdown_contract() -> None:
    messages = design._section_messages(
        prompt="우주 모드를 설계해줘",
        section_id="modules_and_assets",
        fields=("modules", "assets"),
        research={},
    )
    assert "Write design content as Markdown, not JSON" in messages[0]["content"]
    assert (
        "plugin_id | status | reason | requirement_refs | implementation_obligations"
        in messages[0]["content"]
    )
    assert "old three-column module format" in messages[0]["content"]

    parsed = design._module_rows(
        "- mining_core | planning | 광물 채굴 | req_mining, req_economy | "
        "광물 상태 관리; 채굴 보상 계산"
    )
    assert parsed == [
        {
            "plugin_id": "mining_core",
            "status": "planning",
            "reason": "광물 채굴",
            "requirement_refs": ["req_mining", "req_economy"],
            "implementation_obligations": ["광물 상태 관리", "채굴 보상 계산"],
        }
    ]


def test_old_three_column_module_contract_fails_at_parser_boundary() -> None:
    with pytest.raises(SpecValidationError, match="exactly"):
        design._module_rows(
            "- mining_core | planning | 광물 채굴; requirement_refs: req_mining"
        )


def test_requirement_coverage_consumes_the_same_canonical_module_shape() -> None:
    design_value = {
        "modules": [
            {
                "plugin_id": "mining_core",
                "status": "planning",
                "reason": "광물 채굴",
                "requirement_refs": ["req_mining"],
                "implementation_obligations": ["광물 상태 관리"],
            }
        ]
    }
    result = design._validate_requirement_coverage(
        design_value,
        [{"requirement_id": "req_mining"}],
    )
    assert result["_requirement_design_bindings"]["requirement_ids"] == ["req_mining"]
    assert result["_requirement_design_bindings"]["bindings"][0]["module_ids"] == [
        "mining_core"
    ]


def test_unrelated_structured_schema_remains_strict() -> None:
    schema = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
        "required": ["payload"],
        "additionalProperties": False,
    }

    with pytest.raises(StructuredOutputValidationError):
        validate_structured_output(
            '{"payload":["not","an","object"]}',
            response_format="json",
            response_schema=schema,
        )
