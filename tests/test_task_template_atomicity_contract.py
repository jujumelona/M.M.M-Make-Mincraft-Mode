from __future__ import annotations

from pathlib import Path

import yaml

from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema
from minecraft_mod_ai.task_template_catalog import (
    CRITERION_SECTIONS,
    RUNTIME_TEMPLATE_ROOT,
    _materialize_atomic_record_schema,
    load_record_template,
)
from minecraft_mod_ai.worksheet_atomic_chunker import (
    pack_section_concerns,
    worksheet_chunk_schema,
)


def _record_template_ids() -> list[str]:
    identifiers: list[str] = []
    for path in sorted(Path(RUNTIME_TEMPLATE_ROOT).rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "record_schema" not in raw:
            continue
        identifiers.append(
            path.relative_to(RUNTIME_TEMPLATE_ROOT).with_suffix("").as_posix()
        )
    return identifiers


def test_every_runtime_record_template_is_closed_and_model_fillable() -> None:
    identifiers = _record_template_ids()
    assert identifiers, "runtime catalog must contain record templates"

    for identifier in identifiers:
        schema = load_record_template(identifier)["record_schema"]
        assert schema.get("additionalProperties") is False
        assert_atomic_model_schema(schema, surface=f"runtime record {identifier}")


def test_every_model_facing_worksheet_chunk_is_closed() -> None:
    for section in CRITERION_SECTIONS:
        chunks = pack_section_concerns(section)
        assert chunks, f"{section} must compile to at least one model-facing chunk"
        for index, concerns in enumerate(chunks, start=1):
            concern = str(concerns[0])
            schema = worksheet_chunk_schema(
                section,
                concerns,
                include_evidence=index == 1,
                record_counts={concern: 1},
            )
            assert_atomic_model_schema(
                schema,
                surface=f"worksheet chunk {section}[{index}]",
            )


def test_catalog_does_not_invent_global_primitive_bounds() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "tags"],
    }

    compiled = _materialize_atomic_record_schema(schema)

    assert compiled == schema
    assert "maxLength" not in compiled["properties"]["name"]
    assert "maxItems" not in compiled["properties"]["tags"]
    assert "maxLength" not in compiled["properties"]["tags"]["items"]
    assert_atomic_model_schema(compiled, surface="logical record schema")


def test_catalog_preserves_explicit_domain_string_bound_without_global_ceiling() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "maxLength": 4096,
            }
        },
        "required": ["name"],
    }

    compiled = _materialize_atomic_record_schema(schema)
    assert compiled["properties"]["name"]["maxLength"] == 4096
    assert_atomic_model_schema(compiled, surface="explicitly bounded record schema")


def test_catalog_allows_wide_closed_logical_records() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
            "d": {"type": "string"},
            "e": {"type": "string"},
        },
        "required": ["a", "b", "c", "d", "e"],
    }

    compiled = _materialize_atomic_record_schema(schema)
    assert_atomic_model_schema(compiled, surface="wide record schema")
