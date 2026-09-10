from minecraft_mod_ai.task_template_catalog import CRITERION_SECTIONS, detail_records, load_record_template, load_template


def test_criterion_manifests_are_isolated_from_record_templates():
    for section in CRITERION_SECTIONS:
        manifest = load_template(f"criterion/{section}")
        assert manifest["id"] == f"criterion/{section}"
        assert manifest["execution"] == "sequence"
        assert manifest["steps"]


def test_legacy_section_lookup_cannot_shadow_canonical_persistence_record():
    manifest = load_template("feature/persistence")
    record = load_record_template("feature/persistence")
    assert manifest["id"] == "criterion/persistence"
    assert "record_schema" not in manifest
    assert record["id"] == "feature/persistence"
    assert "record_schema" in record


def test_detail_records_are_derived_from_criterion_manifests():
    records = detail_records()
    assert tuple(records) == CRITERION_SECTIONS
    assert "stored_state" in records["persistence"]
