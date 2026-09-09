from __future__ import annotations

"""One-shot source migration from model-authored JSON to fixed tool arguments."""

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "minecraft_mod_ai"
TARGETS = (
    PACKAGE / "central_intelligence_amplifier.py",
    PACKAGE / "complete_orchestrator_services.py",
    PACKAGE / "custom_module_generator.py",
    PACKAGE / "model_smoke.py",
    PACKAGE / "planning_state_implementation.py",
    PACKAGE / "platform_repair_target_contract.py",
    PACKAGE / "repair_engine.py",
    PACKAGE / "resource_asset_production.py",
)
FIXED_IMPORT = "from .fixed_template_generation import generate_fixed_template_text\n"
VALUE_IMPORT = "from .fixed_template_generation import generate_fixed_template_value\n"
CALL_RE = re.compile(
    r"(?P<receiver>\b(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\.generate_text\("
)
JSON_KW_RE = re.compile(r"response_format\s*=\s*(['\"])json\1\s*,?\s*")

PROMPT_REPLACEMENTS = {
    "Return only the compact JSON contract.": "Fill only the supplied compact fixed template.",
    "Return only the JSON contract and no chain-of-thought.": "Fill only the supplied fixed template and no chain-of-thought.",
    "Return only compact JSON and no chain-of-thought.": "Fill only the supplied compact fixed template and no chain-of-thought.",
    "Return the analysis JSON contract, no chain-of-thought.": "Fill the supplied analysis fixed template; no chain-of-thought.",
    "Return exactly one valid JSON object. No markdown.": "Fill the supplied fixed template exactly once. No markdown.",
    "Return exactly {}.": "Fill the supplied empty fixed template.",
    "Return only the JSON object required by the schema.": "Fill only the supplied fixed template.",
    "Return exactly one JSON object with key operations.": "Fill the supplied fixed template with the operations field.",
    "return the final summary in the supplied JSON template.": "fill the final summary in the supplied fixed template.",
    "return inert patch JSON.": "return only inert fixed-template patch fields.",
}


def _add_import(text: str, import_line: str) -> str:
    if import_line.strip() in text:
        return text
    anchor = "from __future__ import annotations\n"
    if anchor not in text:
        raise RuntimeError("missing future-import anchor")
    return text.replace(anchor, anchor + "\n" + import_line, 1)


def _migrate_direct_json_calls(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    count = 0
    for index in range(len(lines)):
        if not JSON_KW_RE.search(lines[index]):
            continue
        start = index
        while start >= 0 and CALL_RE.search(lines[start]) is None:
            start -= 1
        if start < 0:
            raise RuntimeError(f"cannot locate generate_text call for {path}:{index + 1}")
        match = CALL_RE.search(lines[start])
        assert match is not None
        receiver = match.group("receiver")
        lines[start] = CALL_RE.sub(
            f"generate_fixed_template_text({receiver},",
            lines[start],
            count=1,
        )
        lines[index] = JSON_KW_RE.sub("", lines[index], count=1)
        if not lines[index].strip():
            lines[index] = ""
        count += 1

    if count:
        text = "".join(lines)
        text = _add_import(text, FIXED_IMPORT)
        for old, new in PROMPT_REPLACEMENTS.items():
            text = text.replace(old, new)
        ast.parse(text, filename=str(path))
        path.write_text(text, encoding="utf-8")
    return count


def _migrate_criterion_helper() -> None:
    path = PACKAGE / "planning_criterion_fragments.py"
    text = path.read_text(encoding="utf-8")
    old = '''    value = router.generate_tool_decision(\n        "planner",\n        messages,\n        tool_name=_CRITERION_BATCH_TOOL_NAME if batch else _CRITERION_TOOL_NAME,\n        parameters=schema,\n        description=(\n            "Fill the fixed criterion-fragment batch template with exactly one row per supplied criterion_index."\n            if batch\n            else "Fill the fixed criterion-fragment template for the supplied acceptance criterion."\n        ),\n    )\n'''
    new = '''    value = generate_fixed_template_value(\n        router,\n        "planner",\n        messages,\n        response_schema=schema,\n        enable_tools=False,\n        tool_name=_CRITERION_BATCH_TOOL_NAME if batch else _CRITERION_TOOL_NAME,\n        description=(\n            "Fill the fixed criterion-fragment batch template with exactly one row per supplied criterion_index."\n            if batch\n            else "Fill the fixed criterion-fragment template for the supplied acceptance criterion."\n        ),\n    )\n'''
    if new not in text:
        if old not in text:
            raise RuntimeError("criterion fixed-template call site changed unexpectedly")
        text = text.replace(old, new, 1)
    text = _add_import(text, VALUE_IMPORT)
    ast.parse(text, filename=str(path))
    path.write_text(text, encoding="utf-8")


def _direct_json_offenders() -> list[str]:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "generate_text":
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "response_format"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "json"
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return offenders


def main() -> None:
    migrated = sum(_migrate_direct_json_calls(path) for path in TARGETS)
    if migrated != 11:
        raise SystemExit(f"expected to migrate 11 direct JSON calls, migrated {migrated}")
    _migrate_criterion_helper()
    offenders = _direct_json_offenders()
    if offenders:
        raise SystemExit("direct model-authored JSON remains: " + ", ".join(offenders))
    print("migrated 11 direct structured-generation call sites to fixed templates")


if __name__ == "__main__":
    main()
