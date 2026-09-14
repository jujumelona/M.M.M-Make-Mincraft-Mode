from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / ".github" / "scripts" / "audit_code_quality_regression.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("ci_quality_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    # Running a Python script directly places its directory on sys.path. These
    # tests load the script by file location, so reproduce that import contract
    # while keeping the process-wide search path clean after the load.
    script_dir = str(AUDITOR.parent)
    inserted_path = script_dir not in sys.path
    helper_was_loaded = "ci_duplicate_pairs" in sys.modules
    if inserted_path:
        sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_path:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
        if not helper_was_loaded:
            sys.modules.pop("ci_duplicate_pairs", None)
    return module


def test_new_complexity_regression_is_rejected():
    audit = _load_auditor()
    base = audit.analyze_sources({"pkg/mod.py": "def f(x):\n    return x\n"})
    head = audit.analyze_sources(
        {
            "pkg/mod.py": (
                "def f(x):\n"
                "    if x > 0:\n"
                "        if x > 1:\n"
                "            if x > 2:\n"
                "                return 3\n"
                "    return 0\n"
            )
        }
    )
    violations = audit.compare_snapshots(base, head)
    assert any(row["category"] == "complexity_regression" for row in violations)


def test_unchanged_historical_complexity_is_tolerated():
    audit = _load_auditor()
    source = (
        "def old_debt(x):\n"
        "    if x > 0:\n"
        "        if x > 1:\n"
        "            if x > 2:\n"
        "                if x > 3:\n"
        "                    return x\n"
        "    return 0\n"
    )
    snapshot = audit.analyze_sources({"pkg/mod.py": source})
    assert audit.compare_snapshots(snapshot, snapshot) == []


def test_new_import_cycle_is_rejected():
    audit = _load_auditor()
    base = audit.analyze_sources(
        {
            "pkg/a.py": "from pkg import b\n",
            "pkg/b.py": "VALUE = 1\n",
        }
    )
    head = audit.analyze_sources(
        {
            "pkg/a.py": "from pkg import b\n",
            "pkg/b.py": "from pkg import a\n",
        }
    )
    violations = audit.compare_snapshots(base, head)
    assert any(row["category"] == "new_import_cycle" for row in violations)


def test_new_duplicate_function_body_is_rejected():
    audit = _load_auditor()
    base = audit.analyze_sources(
        {
            "pkg/a.py": "def one(x):\n    return x + 1\n",
            "pkg/b.py": "def two(x):\n    return x + 2\n",
        }
    )
    head = audit.analyze_sources(
        {
            "pkg/a.py": "def one(x):\n    return x + 1\n",
            "pkg/b.py": "def two(x):\n    return x + 1\n",
        }
    )
    violations = audit.compare_snapshots(base, head)
    assert any(row["category"] == "new_duplicate_function_body" for row in violations)


def test_new_serial_expensive_loop_is_rejected():
    audit = _load_auditor()
    base = audit.analyze_sources({"pkg/a.py": "def run(items, model):\n    return []\n"})
    head = audit.analyze_sources(
        {
            "pkg/a.py": (
                "def run(items, model):\n"
                "    out = []\n"
                "    for item in items:\n"
                "        out.append(model.generate(item))\n"
                "    return out\n"
            )
        }
    )
    violations = audit.compare_snapshots(base, head)
    assert any(row["category"] == "new_serial_expensive_loop" for row in violations)


def test_main_ci_topology_requires_full_suite_and_authoritative_audits():
    audit = _load_auditor()
    good = """
    python .github/scripts/audit_runtime_concurrency.py
    python .github/scripts/audit_runtime_efficiency.py --output audit/runtime-efficiency.json
    python .github/scripts/audit_code_quality_regression.py --output audit/code-quality-regression.json
    python tools/verify_integrity_minecraft.py --output .mmm/integrity-validation
    python tools/root_cause_audit_wrapper.py
    python tools/pytest_diagnostics.py tests
    needs: [audit, tests, python313, model-realistic-replay]
    """
    assert audit.audit_main_ci_text(good) == []

    bad = good.replace("python tools/pytest_diagnostics.py tests", "python -m pytest tests/test_one.py")
    bad = bad.replace("audit_runtime_concurrency.py", "missing_concurrency.py")
    violations = audit.audit_main_ci_text(bad)
    categories = {row["category"] for row in violations}
    assert "ci_missing_full_test_suite" in categories
    assert "ci_missing_authoritative_audit" in categories


def test_new_function_hard_ceiling_is_rejected():
    audit = _load_auditor()
    branches = "".join(f"    if x == {index}:\n        return {index}\n" for index in range(20))
    head = audit.analyze_sources({"pkg/a.py": "def pathological(x):\n" + branches + "    return -1\n"})
    violations = audit.compare_snapshots(audit.analyze_sources({}), head)
    assert any(row["category"] == "new_function_hard_ceiling" for row in violations)
