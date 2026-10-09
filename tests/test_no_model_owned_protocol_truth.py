"""Regression tests for the *current* host-owned model/execution boundaries.

The retired model-size and arbitrary-Java pipelines must not be resurrected.
Test semantic invariants through the active authority rather than reading
modules deleted by the deterministic backend migration.
"""
from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.execution_contract_policy import ATOMIC_REGION_COMPLETION_PARAMETERS
from minecraft_mod_ai.bounded_record_template import record_cardinality_response_schema
from minecraft_mod_ai.worksheet_atomic_chunker import (
    pack_section_concerns,
    worksheet_chunk_schema,
    worksheet_concern_cardinality_schema,
)

PACKAGE = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def test_record_generation_has_host_bounded_cardinality_without_model_done_protocol() -> None:
    bounded = _source("bounded_record_template.py")
    design = _source("design_record_runtime.py")
    schema = record_cardinality_response_schema(minimum_count=0)
    assert schema["required"] == ["count"]
    assert schema["properties"]["count"]["minimum"] == 0
    assert "done" not in schema["properties"]
    assert "record_cardinality_response_schema" in bounded
    assert "submit_next_" not in bounded
    assert '"record": None' not in bounded
    assert "design/content_entity_count" not in design
    assert "design/content_relation_count" not in design
    assert not (PACKAGE / "templates/design/content_entity_count.yaml").exists()
    assert not (PACKAGE / "templates/design/content_relation_count.yaml").exists()


def test_worksheet_uses_host_fixed_page_cardinality_and_field_projections() -> None:
    schema = worksheet_concern_cardinality_schema("behavior_contract", "actors")
    assert schema["required"] == ["record_count"]
    assert schema["properties"]["record_count"]["type"] == "integer"
    page = pack_section_concerns("behavior_contract")[0]
    projection = page.field_projection["actors"]
    assert projection
    fixed = worksheet_chunk_schema(
        "behavior_contract", page, record_counts={"actors": 1},
    )
    actors = fixed["properties"]["actors"]
    assert actors["minItems"] == actors["maxItems"] == 1
    assert actors["items"]["required"] == list(projection)
    assert "record_counts.setdefault" not in _source("worksheet_atomic_chunker.py")


def test_structured_output_is_schema_bounded_not_obsolete_global_field_depth_gate() -> None:
    boundary = _source("model_output_atomicity_contract.py")
    catalog = _source("task_template_catalog.py")
    single = _source("single_record_template.py")
    assert "assert_atomic_model_schema" in boundary
    assert "structured_output_token_ceiling" in boundary
    for marker in (
        "MODEL_ATOMICITY_FIELDS_EXCEEDED",
        "MODEL_ATOMICITY_DEPTH_EXCEEDED",
        "MODEL_ATOMICITY_ARRAY_EXCEEDED",
        "MODEL_ATOMICITY_STRING_EXCEEDED",
    ):
        assert marker not in boundary
    assert 'setdefault("maxLength", MAX_MODEL_STRING_CHARS)' not in catalog
    assert 'setdefault("maxItems", MAX_MODEL_ARRAY_ITEMS)' not in catalog
    assert "MAX_MODEL_FIELDS" not in single


def test_java_semantic_completion_has_no_retired_model_owned_paging_runtime() -> None:
    schema = ATOMIC_REGION_COMPLETION_PARAMETERS
    assert schema["required"] == ["next_work"]
    assert "maxLength" not in schema["properties"]["next_work"]
    assert "done" in schema["properties"]
    assert not (PACKAGE / "atomic_region_paging.py").exists()
    assert not (PACKAGE / "atomic_region_work.py").exists()


def test_harmless_json_wrapping_is_host_recoverable_but_schema_truth_remains_strict() -> None:
    structured = _source("structured_output.py")
    assert "_extract_schema_valid_embedded_value" in structured
    assert "len(valid) == 1" in structured


def test_design_text_is_not_rejected_by_phrase_blacklist() -> None:
    meta = _source("model_meta_output_contract.py")
    guarded = meta.split("def assert_design_field_clean", 1)[1]
    assert "contains_internal_model_meta(value)" not in guarded
    assert "contradicts_grounded_research(value)" in guarded


def test_executable_semantics_do_not_depend_on_legacy_model_size_aliases() -> None:
    # Transport/decoder budget owners are allowed finite limits; correctness
    # must not be predicated on one model's arbitrary field/depth cap.
    production = (
        "design_record_runtime.py",
        "typed_plan_production.py",
        "artifact_graph_executor.py",
        "work_graph.py",
    )
    old_semantic_caps = (
        "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_fields",
        "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_schema_depth",
        "MAX_MODEL_FIELDS",
        "MAX_MODEL_ARRAY_ITEMS",
    )
    for module in production:
        source = _source(module)
        for token in old_semantic_caps:
            assert token not in source, f"{module}: {token}"


def test_retired_arbitrary_java_generator_does_not_bypass_typed_backend() -> None:
    for retired in (
        "custom_module_generator.py",
        "atomic_java_assembly.py",
        "atomic_java_admission.py",
        "atomic_concern_source.py",
    ):
        assert not (PACKAGE / retired).exists(), retired
    work_graph = _source("work_graph.py")
    typed = _source("typed_plan_production.py")
    assert "DETERMINISTIC_BACKEND_REQUIRED" in work_graph
    assert "typed_host" in work_graph
    assert "typed_host" in typed
