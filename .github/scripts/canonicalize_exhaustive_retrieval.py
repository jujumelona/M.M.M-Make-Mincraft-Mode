from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path.cwd()
GH = ROOT / "minecraft_mod_ai" / "github_adaptive_retrieval.py"
RG = ROOT / "minecraft_mod_ai" / "research_grounded_rag_contract.py"


def _remove_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    lines = text.splitlines(keepends=True)
    for start, end in sorted(ranges, reverse=True):
        del lines[start - 1 : end]
    return "".join(lines)


def _remove_shadowed_top_level_functions(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    defs: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, []).append(node)
    ranges: list[tuple[int, int]] = []
    for nodes in defs.values():
        for node in nodes[:-1]:
            ranges.append((node.lineno, node.end_lineno or node.lineno))
    if ranges:
        path.write_text(_remove_ranges(text, ranges), encoding="utf-8")


def _remove_named_functions(path: Path, names: set[str]) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            ranges.append((node.lineno, node.end_lineno or node.lineno))
    if ranges:
        path.write_text(_remove_ranges(text, ranges), encoding="utf-8")


def _remove_obsolete_assignments(path: Path, names: set[str]) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        assigned: set[str] = set()
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
        if assigned & names:
            ranges.append((node.lineno, node.end_lineno or node.lineno))
    if ranges:
        path.write_text(_remove_ranges(text, ranges), encoding="utf-8")


# The migration first appends tested replacements so it cannot damage the original on
# partial failure.  Canonicalization then removes every shadowed implementation.  This
# means the committed module contains one executable definition per retrieval function,
# not a legacy numeric-budget implementation hidden above an override.
_remove_shadowed_top_level_functions(GH)
_remove_shadowed_top_level_functions(RG)

# These helpers existed only to terminate discovery using locally invented numeric
# thresholds.  They are not resource-safety mechanisms after the frontier contract is
# installed, so remove them completely rather than leaving dead policy in the module.
_remove_named_functions(
    GH,
    {
        "_env_float",
        "_search_request_budget",
        "_search_page_size",
        "_source_request_budget",
        "_source_byte_budget",
        "_output_byte_budget",
        "_coverage_target",
    },
)

_remove_obsolete_assignments(
    RG,
    {
        "_MAX_SOURCE_TEXT_CHARS",
        "_MAX_EXTERNAL_PROJECTS",
    },
)
