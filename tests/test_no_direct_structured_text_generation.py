from __future__ import annotations

import ast
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"


def _direct_json_generate_text_calls() -> list[str]:
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "generate_text":
                continue
            for keyword in node.keywords:
                if keyword.arg != "response_format":
                    continue
                if (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "json"
                ):
                    offenders.append(
                        f"{path.relative_to(_PACKAGE_ROOT.parent)}:{node.lineno}"
                    )
    return offenders


def test_production_code_never_requests_model_authored_json_text():
    offenders = _direct_json_generate_text_calls()
    assert offenders == [], (
        "Model-authored JSON text is forbidden. Use generate_tool_decision() with a closed "
        "fixed argument template instead. Offenders: " + ", ".join(offenders)
    )
