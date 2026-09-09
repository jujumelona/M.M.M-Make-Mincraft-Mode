from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "audit_runtime_mutations.py"
SPEC = importlib.util.spec_from_file_location("runtime_mutation_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _finding(*, line: int, target: str = "runtime.patch") -> dict:
    return {
        "path": "minecraft_mod_ai/runtime_contract.py",
        "line": line,
        "scope": "install",
        "kind": "external_attribute_rebind",
        "target": target,
        "metadata_only": False,
    }


def test_runtime_mutation_delta_ignores_code_motion_but_preserves_multiplicity() -> None:
    baseline = [_finding(line=10)]
    moved = [_finding(line=200)]
    introduced, removed = AUDIT.compare_behavioral_findings(moved, baseline)

    assert introduced == []
    assert removed == []

    duplicated = [_finding(line=200), _finding(line=201)]
    introduced, removed = AUDIT.compare_behavioral_findings(duplicated, baseline)

    assert len(introduced) == 1
    assert introduced[0]["line"] == 201
    assert removed == []


def test_runtime_mutation_delta_audits_parent_and_current_with_same_scanner(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    current_root = tmp_path / "current"
    baseline_package = baseline_root / "minecraft_mod_ai"
    current_package = current_root / "minecraft_mod_ai"
    baseline_package.mkdir(parents=True)
    current_package.mkdir(parents=True)

    baseline_source = "import vendor\nvendor.one = replacement\n"
    current_source = "import vendor\n\nvendor.one = replacement\nvendor.two = replacement\n"
    (baseline_package / "contract.py").write_text(baseline_source, encoding="utf-8")
    (current_package / "contract.py").write_text(current_source, encoding="utf-8")

    baseline = AUDIT.audit(baseline_root)
    current = AUDIT.audit(current_root)
    introduced, removed = AUDIT.compare_behavioral_findings(current, baseline)

    assert [item["target"] for item in introduced] == ["vendor.two"]
    assert removed == []
