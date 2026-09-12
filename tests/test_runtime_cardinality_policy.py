from minecraft_mod_ai.task_template_catalog import (
    CRITERION_SECTIONS,
    load_record_template,
    load_template,
)


EVIDENCE_BOUND_CARDINALITY_TASKS = frozenset(
    {
        "feature/integration/target_bindings",
        "feature/reuse_assessment/candidates",
        "feature/reuse_assessment/compatibility",
        "feature/reuse_assessment/conflicts",
        "feature/reuse_assessment/gaps",
        "feature/reuse_assessment/license",
        "feature/reuse_assessment/mapping",
        "feature/reuse_assessment/selection",
        "feature/reuse_assessment/sources",
        "feature/reuse_assessment/verification",
    }
)


def _runtime_concern_ids():
    identifiers = []
    for section in CRITERION_SECTIONS:
        manifest = load_template(f"criterion/{section}")
        identifiers.extend(manifest["steps"])
    return identifiers


def test_runtime_cardinality_policy_is_total_and_exact():
    identifiers = _runtime_concern_ids()
    assert len(identifiers) == 83
    assert len(identifiers) == len(set(identifiers))
    assert EVIDENCE_BOUND_CARDINALITY_TASKS <= set(identifiers)

    semantic = set(identifiers) - EVIDENCE_BOUND_CARDINALITY_TASKS
    assert len(semantic) == 73
    assert len(EVIDENCE_BOUND_CARDINALITY_TASKS) == 10

    for identifier in identifiers:
        template = load_record_template(identifier)
        expected = identifier in EVIDENCE_BOUND_CARDINALITY_TASKS
        assert template["cardinality_blocking"] is expected, identifier


def test_runtime_record_templates_keep_host_control_out_of_model_rules():
    forbidden = ("return done", "return not_applicable", "return blocked")
    for identifier in _runtime_concern_ids():
        template = load_record_template(identifier)
        rules = "\n".join(str(rule).lower() for rule in template.get("rules", ()))
        for phrase in forbidden:
            assert phrase not in rules, f"{identifier}: {phrase}"
        assert "host owns cardinality and iteration" in rules, identifier
