from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
REMOVED_SYMBOL = "generate_fixed_template_" + "text"
CURRENT_SYMBOL = "generate_fixed_template_value"


def test_removed_fixed_template_text_api_has_no_runtime_consumers() -> None:
    offenders: list[str] = []
    for source in sorted(PACKAGE.rglob("*.py")):
        text = source.read_text(encoding="utf-8")
        if REMOVED_SYMBOL in text:
            offenders.append(str(source.relative_to(ROOT)))

    assert offenders == [], (
        "Removed fixed-template text API still has runtime consumers: "
        + ", ".join(offenders)
    )


def test_fixed_template_generation_exports_only_current_value_api() -> None:
    source = PACKAGE / "fixed_template_generation.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    exported: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exported.update(
                        element.value
                        for element in node.value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    )

    assert CURRENT_SYMBOL in exported
    assert REMOVED_SYMBOL not in exported
