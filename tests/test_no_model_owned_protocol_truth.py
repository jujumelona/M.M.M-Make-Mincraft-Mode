from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.execution_contract_policy import ATOMIC_REGION_COMPLETION_PARAMETERS


PACKAGE = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def test_record_generation_has_no_model_owned_cardinality_prepass() -> None:
    bounded = _source("bounded_record_template.py")
    design = _source("design_record_runtime.py")

    assert "_load_cardinality" not in bounded
    assert "_count" not in bounded
    assert '"record_count": count' not in bounded
    assert "TEMPLATE_NO_PROGRESS: repeated record" not in bounded
    assert "submit_next_" not in bounded
    assert '"record": None' not in bounded
    assert '"done"' not in bounded
    assert "design/content_entity_count" not in design
    assert "design/content_relation_count" not in design
    assert not (PACKAGE / "templates/design/content_entity_count.yaml").exists()
    assert not (PACKAGE / "templates/design/content_relation_count.yaml").exists()


def test_worksheet_does_not_use_model_count_or_array_index_as_cross_call_identity() -> None:
    chunker = _source("worksheet_atomic_chunker.py")
    planner = _source("planning_state_implementation.py")

    assert "DETAILED_PLAN_WORKSHEET_FIELD_PAGE_COUNT" not in chunker
    assert "Host-fixed Record Counts" not in chunker
    assert "record_counts.setdefault" not in planner
    assert "record_counts: dict" not in planner


def test_structured_output_correctness_has_no_global_3_4_3_256_size_gate() -> None:
    boundary = _source("model_output_atomicity_contract.py")
    catalog = _source("task_template_catalog.py")
    single = _source("single_record_template.py")
    assembly = _source("atomic_java_assembly.py")
    central = _source("central_atomic_generation_contract.py")

    for obsolete_failure in (
        "MODEL_ATOMICITY_FIELDS_EXCEEDED",
        "MODEL_ATOMICITY_DEPTH_EXCEEDED",
        "MODEL_ATOMICITY_ARRAY_EXCEEDED",
        "MODEL_ATOMICITY_STRING_EXCEEDED",
        "MODEL_ATOMICITY_ARRAY_UNBOUNDED",
        "MODEL_ATOMICITY_STRING_UNBOUNDED",
    ):
        assert obsolete_failure not in boundary

    assert 'setdefault("maxLength", MAX_MODEL_STRING_CHARS)' not in catalog
    assert 'setdefault("maxItems", MAX_MODEL_ARRAY_ITEMS)' not in catalog
    assert "MAX_MODEL_FIELDS" not in single
    assert "DEFAULT_ATOMIC_SCHEMA_LIMITS" not in assembly
    assert "DEFAULT_ATOMIC_SCHEMA_LIMITS" not in central
    verifier = _source("verifier_repair_window.py")
    assert "DEFAULT_ATOMIC_SCHEMA_LIMITS" not in verifier


def test_java_paging_completion_uses_semantic_next_work_only() -> None:
    paging = _source("atomic_region_paging.py")
    schema = ATOMIC_REGION_COMPLETION_PARAMETERS

    assert schema["required"] == ["next_work"]
    assert "maxLength" not in schema["properties"]["next_work"]
    assert "done" in schema["properties"]  # optional compatibility hint only
    assert "set(decision)" not in paging
    assert "len(decision[\"next_work\"])" not in paging
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


def test_no_production_module_consumes_legacy_model_size_aliases() -> None:
    forbidden = (
        "MAX_MODEL_FIELDS",
        "MAX_MODEL_STRING_CHARS",
        "MAX_MODEL_ARRAY_ITEMS",
        "MAX_SCHEMA_DEPTH",
        "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_fields",
        "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_string_chars",
        "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items",
        "DEFAULT_ATOMIC_SCHEMA_LIMITS.max_schema_depth",
    )
    definition_owners = {
        "execution_contract_policy.py",
        "model_output_atomicity_contract.py",
    }
    offenders: list[str] = []
    for candidate in sorted(PACKAGE.rglob("*.py")):
        if candidate.name in definition_owners:
            continue
        source = candidate.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{candidate.relative_to(PACKAGE)}: {token}")
    assert not offenders, "legacy model-size correctness coupling remains:\n" + "\n".join(offenders)


def test_atomic_java_production_has_one_structured_materialization_path() -> None:
    generator = _source("custom_module_generator.py")
    concern = _source("atomic_concern_source.py")

    atomic_start = generator.index("def _run_atomic_ir_generation(")
    atomic_end = generator.index("\nclass CustomModuleGenerator", atomic_start)
    atomic_path = generator[atomic_start:atomic_end]

    assert "_call_atomic_java_region(" in atomic_path
    assert "_call_coder(" not in atomic_path
    assert "_production_atomic_coder" not in generator
    assert "structured_java_region" not in generator
    assert "using direct source" not in generator

    assert "admit_member_region" not in concern
    assert "admit_initialize_region" not in concern
    assert "parse_concern_content" not in concern
    assert "<<<MMM_CONCERN_MEMBERS>>>" not in concern
    assert "<<<MMM_CONCERN_INITIALIZE>>>" not in concern


def test_atomic_structured_router_owns_java_syntax_and_identifier_rendering() -> None:
    generator = _source("custom_module_generator.py")
    policy = _source("execution_contract_policy.py")

    assert "def _call_atomic_java_region(" in generator
    assert "JavaStructureAssembly(" in generator
    assert "_render_atomic_java_structure(" in generator
    assert "_java_identifier_renames(" in generator
    assert "Semantic field name" in policy
    assert "host owns Java identifier spelling" in policy


def test_atomic_java_names_are_host_materialized_not_model_rejected() -> None:
    assembly = _source("atomic_java_assembly.py")
    policy = _source("execution_contract_policy.py")
    generator = _source("custom_module_generator.py")

    assert 'scalars["name"]["not"]' not in assembly
    assert 'item_scalars["name"]["not"]' not in assembly
    assert "duplicate or reserved" not in assembly
    assert 'name_schema["not"]' not in policy
    assert "def _host_identifier_overrides(" in generator
    assert "reserved_identifiers=reserved_type_names" in generator
    assert "preferred_identifiers=identifier_overrides" in generator
