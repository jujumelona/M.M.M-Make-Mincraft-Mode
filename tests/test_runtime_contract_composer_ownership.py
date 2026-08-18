from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
_APPROVED = {
    "runtime_bootstrap.py",
    "llama_tuning_pipeline.py",
}


def _composer_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "compose_contract_stages":
            calls.append(node.lineno)
            continue
        if isinstance(func, ast.Attribute) and func.attr == "compose_contract_stages":
            calls.append(node.lineno)
    return calls


def test_only_explicit_runtime_owners_may_invoke_contract_composer() -> None:
    actual: dict[str, list[int]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        calls = _composer_calls(path)
        if calls:
            actual[path.name] = calls

    assert set(actual) == _APPROVED, (
        "contract composer ownership changed; add composition only through the "
        f"approved runtime owners: {actual}"
    )


def test_package_bootstrap_phase_order_is_explicit_and_stable() -> None:
    source = (PACKAGE / "runtime_bootstrap.py").read_text(encoding="utf-8")
    order = (
        'ContractStage("prebootstrap"',
        'ContractStage("core"',
        'ContractStage(\n                "model-runtime"',
        'ContractStage("validation"',
        'ContractStage("generation"',
        'ContractStage("platform"',
        'ContractStage("planner"',
        'ContractStage(\n                "architecture"',
        'ContractStage("late-safety"',
        'ContractStage(\n                "public-boundary"',
        'ContractStage(\n                "post-bootstrap"',
        'ContractStage("postbootstrap"',
    )
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "state_owner=_contract_composer" in source


def test_llama_pipeline_uses_same_fail_closed_kernel() -> None:
    source = (PACKAGE / "llama_tuning_pipeline.py").read_text(encoding="utf-8")
    assert 'owner_name="native-llama-tuning"' in source
    assert "compose_contract_stages(" in source
    assert "boundaries=self._callable_boundaries()" in source
    assert "_mmm_tuning_pipeline_receipts" in source
