from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(".github/scripts/audit_code_quality_regression.py")
SPEC = importlib.util.spec_from_file_location("audit_code_quality_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _workflow(needs: str) -> str:
    required_tokens = "\n".join(
        (
            "pytest_diagnostics.py tests",
            "audit_runtime_concurrency.py",
            "audit_runtime_efficiency.py",
            "audit_code_quality_regression.py",
            "verify_integrity_minecraft.py",
            "root_cause_audit_wrapper.py",
        )
    )
    return f"""jobs:
  ci-gate:
    if: ${{{{ always() }}}}
    needs: {needs}
    steps:
      - run: echo ok
{required_tokens}
"""


def _categories(text: str) -> set[str]:
    return {row["category"] for row in AUDIT.audit_main_ci_text(text)}


def test_ci_gate_dependency_audit_requires_model_replay() -> None:
    categories = _categories(_workflow("[audit, tests, python313]"))
    assert "ci_gate_dependency_gap" in categories


def test_ci_gate_dependency_audit_accepts_required_dependencies() -> None:
    categories = _categories(
        _workflow("[audit, tests, python313, model-realistic-replay]")
    )
    assert "ci_gate_dependency_gap" not in categories


def test_ci_gate_dependency_audit_accepts_future_extra_dependencies() -> None:
    categories = _categories(
        _workflow(
            "[audit, tests, python313, model-realistic-replay, future-required-gate]"
        )
    )
    assert "ci_gate_dependency_gap" not in categories
