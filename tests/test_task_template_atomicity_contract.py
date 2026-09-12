from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.model_output_atomicity_contract import (
    MAX_MODEL_ARRAY_ITEMS,
    MAX_MODEL_STRING_CHARS,
    assert_atomic_model_schema,
)
from minecraft_mod_ai.task_template_catalog import (
    RUNTIME_TEMPLATE_ROOT,
    _materialize_atomic_record_schema,
    load_record_template,
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


def test_every_runtime_record_template_compiles_to_atomic_model_schema() -> None:
    identifiers = _record_template_ids()
    assert identifiers, "runtime catalog must contain record templates"

    for identifier in identifiers:
        template = load_record_template(identifier)
        assert_atomic_model_schema(
            template["record_schema"],
            surface=f"test runtime record template {identifier!r}",
        )


def test_catalog_materializes_only_missing_global_primitive_bounds() -> None:
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

    assert "maxLength" not in schema["properties"]["name"]
    assert compiled["properties"]["name"]["maxLength"] == MAX_MODEL_STRING_CHARS
    assert compiled["properties"]["tags"]["maxItems"] == MAX_MODEL_ARRAY_ITEMS
    assert (
        compiled["properties"]["tags"]["items"]["maxLength"]
        == MAX_MODEL_STRING_CHARS
    )
    assert_atomic_model_schema(compiled, surface="compiled record schema")


def test_catalog_does_not_relax_explicit_oversized_string_bound() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "maxLength": MAX_MODEL_STRING_CHARS + 1,
            }
        },
        "required": ["name"],
    }

    compiled = _materialize_atomic_record_schema(schema)
    assert compiled["properties"]["name"]["maxLength"] == MAX_MODEL_STRING_CHARS + 1
    with pytest.raises(ModelConfigurationError, match="MODEL_ATOMICITY_STRING_EXCEEDED"):
        assert_atomic_model_schema(compiled, surface="oversized record schema")


def test_catalog_does_not_relax_structural_field_limit() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
            "d": {"type": "string"},
        },
        "required": ["a", "b", "c", "d"],
    }

    compiled = _materialize_atomic_record_schema(schema)
    with pytest.raises(ModelConfigurationError, match="MODEL_ATOMICITY_FIELDS_EXCEEDED"):
        assert_atomic_model_schema(compiled, surface="wide record schema")
