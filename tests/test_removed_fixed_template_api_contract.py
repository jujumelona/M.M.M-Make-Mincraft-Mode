from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


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


def test_colab_bootstrap_import_chain_loads_in_clean_interpreter() -> None:
    code = (
        "import minecraft_mod_ai; "
        "import minecraft_mod_ai.jdtls_bootstrap; "
        "import minecraft_mod_ai.complete_orchestrator_services; "
        "import minecraft_mod_ai.model_smoke; "
        "import minecraft_mod_ai.central_intelligence_amplifier"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        "Clean-process Colab bootstrap import chain failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
