from pathlib import Path

import yaml

from minecraft_mod_ai.feature_template_pipeline import ATOMIC_CHECKS, atomic_leaves, evaluate_atomicity


def _checks(failed=()):
    failed = set(failed)
    return [{"check": name, "passed": name not in failed, "reason": "explicit test evidence"} for name in ATOMIC_CHECKS]


def test_host_derives_atomicity_from_every_required_check():
    result = evaluate_atomicity(_checks())
    assert result["atomic"] is True
    assert result["failed_checks"] == []
    result = evaluate_atomicity(_checks({"explicit_trigger", "test_case_writable"}))
    assert result["atomic"] is False
    assert result["failed_checks"] == ["explicit_trigger", "test_case_writable"]


def test_atomicity_rejects_missing_or_duplicate_checks():
    records = _checks()
    try:
        evaluate_atomicity(records[:-1])
    except ValueError as exc:
        assert "missing checks" in str(exc)
    else:
        raise AssertionError("missing atomic check was accepted")
    try:
        evaluate_atomicity(records + [records[0]])
    except ValueError as exc:
        assert "duplicate check" in str(exc)
    else:
        raise AssertionError("duplicate atomic check was accepted")


def test_atomic_leaves_rejects_unresolved_non_atomic_node():
    tree = {"atomicity": {"atomic": False}, "children": []}
    try:
        list(atomic_leaves(tree))
    except ValueError as exc:
        assert "must have children" in str(exc)
    else:
        raise AssertionError("unresolved non-atomic feature was accepted")


def test_atomic_template_never_owns_final_atomic_decision():
    path = Path(__file__).parents[1] / "minecraft_mod_ai" / "templates" / "feature" / "atomic_check.yaml"
    template = yaml.safe_load(path.read_text(encoding="utf-8"))
    props = template["record_schema"]["properties"]
    assert "atomic" not in props
    assert set(props) == {"check", "passed", "reason"}
    assert tuple(props["check"]["enum"]) == ATOMIC_CHECKS
