from minecraft_mod_ai.task_template_catalog import (
    CRITERION_SECTIONS,
    load_record_template,
    load_template,
)


def _runtime_concern_ids():
    identifiers = []
    for section in CRITERION_SECTIONS:
        manifest = load_template(f"criterion/{section}")
        identifiers.extend(manifest["steps"])
    return identifiers


def test_runtime_cardinality_metadata_is_not_model_visible():
    identifiers = _runtime_concern_ids()
    assert len(identifiers) == 83
    assert len(identifiers) == len(set(identifiers))

    for identifier in identifiers:
        template = load_record_template(identifier)
        assert "cardinality_blocking" not in template, identifier


def test_runtime_record_templates_forbid_model_loop_control():
    forbidden = (
        "return done",
        "return not_applicable",
        "return blocked",
        "host owns cardinality",
        "host-requested ordinal",
    )
    for identifier in _runtime_concern_ids():
        template = load_record_template(identifier)
        rules = "\n".join(str(rule).lower() for rule in template.get("rules", ()))
        for phrase in forbidden:
            assert phrase not in rules, f"{identifier}: {phrase}"
        assert "loop-control protocol" in rules, identifier
