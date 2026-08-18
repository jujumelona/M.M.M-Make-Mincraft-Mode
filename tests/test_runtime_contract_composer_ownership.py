from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
_APPROVED = {
    "runtime_bootstrap.py",
    "llama_tuning_pipeline.py",
}
_COMPOSER_MODULE = "runtime_contract_composer"
_COMPOSER_FUNCTION = "compose_contract_stages"


def _is_composer_module(name: str | None) -> bool:
    if not name:
        return False
    return name.rsplit(".", 1)[-1] == _COMPOSER_MODULE


def _composer_references(path: Path) -> list[int]:
    """Find composer imports/references, not only calls, so aliases cannot bypass ownership."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _is_composer_module(node.module):
                lines.add(node.lineno)
                continue
            if node.level and node.module is None and any(
                imported.name == _COMPOSER_MODULE for imported in node.names
            ):
                lines.add(node.lineno)
                continue
        if isinstance(node, ast.Import):
            if any(_is_composer_module(imported.name) for imported in node.names):
                lines.add(node.lineno)
                continue
        if isinstance(node, ast.Name) and node.id == _COMPOSER_FUNCTION:
            lines.add(node.lineno)
            continue
        if isinstance(node, ast.Attribute) and node.attr == _COMPOSER_FUNCTION:
            lines.add(node.lineno)
    return sorted(lines)


def test_only_explicit_runtime_owners_may_reference_contract_composer() -> None:
    actual: dict[str, list[int]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        references = _composer_references(path)
        if references:
            actual[path.name] = references

    assert set(actual) == _APPROVED, (
        "contract composer ownership changed; import/reference composition only "
        f"through the approved runtime owners: {actual}"
    )


def test_aliases_cannot_hide_composer_ownership(tmp_path: Path) -> None:
    direct_alias = tmp_path / "direct_alias.py"
    direct_alias.write_text(
        "from minecraft_mod_ai.runtime_contract_composer import "
        "compose_contract_stages as compose\n"
        "compose(owner_name='x', version=1, state_owner=object(), stages=())\n",
        encoding="utf-8",
    )
    module_alias = tmp_path / "module_alias.py"
    module_alias.write_text(
        "import minecraft_mod_ai.runtime_contract_composer as runtime_contract\n"
        "compose = runtime_contract.compose_contract_stages\n"
        "compose(owner_name='x', version=1, state_owner=object(), stages=())\n",
        encoding="utf-8",
    )

    assert _composer_references(direct_alias)
    assert _composer_references(module_alias)


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


def test_llama_runtime_types_are_owned_before_any_tuning_wrapper() -> None:
    source = (PACKAGE / "llama_tuning_pipeline.py").read_text(encoding="utf-8")
    runtime_types = source.index('TuningStage("runtime-types"')
    hardware = source.index('TuningStage("hardware"')
    runtime = source.index('TuningStage("runtime"')

    assert runtime_types < hardware < runtime
    assert "_install_runtime_type_ownership" in source
    assert "_mmm_runtime_tuning_type_owner" in source
    assert '"autotune.server_variant"' in source
    assert '"ubatch"' in source
    assert '"parallel"' in source
    assert '"cache_reuse"' in source
    assert '"draft_p_min"' in source
