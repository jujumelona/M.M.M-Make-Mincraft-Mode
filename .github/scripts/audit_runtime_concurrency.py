from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (ROOT / "minecraft_mod_ai",)

# Direct executor ownership is an architectural capability, not a convenience import.
# These modules already own a distinct scheduler/resource lifecycle (DAG scheduling,
# commit serialization, CPU extraction, native-runtime tuning, etc.) and are therefore
# explicit reviewed owners. New modules are fail-closed: independent fan-out must use
# deadline_executor instead of growing another scheduler. Remove entries from this set
# as legacy owners are migrated; never add a caller merely to make this audit green.
REVIEWED_EXECUTOR_OWNERS = {
    Path("minecraft_mod_ai/agentic_optimization_contract.py"),
    Path("minecraft_mod_ai/agentic_search_efficiency_contract.py"),
    Path("minecraft_mod_ai/api_symbol_extractor.py"),
    Path("minecraft_mod_ai/artifact_graph_executor.py"),
    Path("minecraft_mod_ai/bounded_record_template.py"),
    Path("minecraft_mod_ai/coder_max_efficiency_contract.py"),
    Path("minecraft_mod_ai/complete_orchestrator.py"),
    Path("minecraft_mod_ai/deadline_executor.py"),
    Path("minecraft_mod_ai/generation_concurrency_safety.py"),
    Path("minecraft_mod_ai/llama_server_runtime_tuning.py"),
    Path("minecraft_mod_ai/parallel_runtime_contract.py"),
    Path("minecraft_mod_ai/planning_state_implementation.py"),
    Path("minecraft_mod_ai/pre_design_external_source_contract.py"),
    Path("minecraft_mod_ai/research_grounded_rag_contract.py"),
    Path("minecraft_mod_ai/research_version_catalog.py"),
    Path("minecraft_mod_ai/runtime_regression_reconciliation.py"),
    Path("minecraft_mod_ai/scalable_pipeline.py"),
    Path("minecraft_mod_ai/source_patch.py"),
}
EXECUTOR_TYPES = {"ThreadPoolExecutor", "ProcessPoolExecutor"}

PATTERNS = {
    "executor": re.compile(r"\b(?:ThreadPoolExecutor|ProcessPoolExecutor)\b"),
    "future_result": re.compile(r"\.result\s*\("),
    "shutdown_wait_true": re.compile(r"\.shutdown\s*\(\s*wait\s*=\s*True"),
    "executor_map": re.compile(r"\.(?:map)\s*\("),
    "as_completed": re.compile(r"\bas_completed\s*\("),
    "wait_call": re.compile(r"\bwait\s*\("),
    "tool_round_cap": re.compile(
        r"(?:_agent_tool_round_limit|MMM_AGENT_TOOL_ROUNDS|DEFAULT_AGENT_TOOL_ROUNDS|round_limit)"
    ),
    "candidate_score_constant": re.compile(
        r"(?:1000\.0|candidate_count|scored\.append)"
    ),
}


def source_files() -> list[Path]:
    return sorted(path for root in SCAN_ROOTS for path in root.rglob("*.py"))


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _executor_import_ownership(tree: ast.AST) -> tuple[set[str], set[str], set[tuple[int, str]]]:
    imported_aliases: set[str] = set()
    module_aliases: set[str] = set()
    findings: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "concurrent.futures":
            for alias in node.names:
                if alias.name in EXECUTOR_TYPES:
                    local_name = alias.asname or alias.name
                    imported_aliases.add(local_name)
                    findings.add((node.lineno, f"imports {alias.name} as {local_name}"))
        elif isinstance(node, ast.Import):
            module_aliases.update(
                alias.asname or "concurrent.futures"
                for alias in node.names
                if alias.name == "concurrent.futures"
            )
    return imported_aliases, module_aliases, findings


def _executor_construction_findings(
    tree: ast.AST, imported_aliases: set[str], module_aliases: set[str]
) -> set[tuple[int, str]]:
    findings: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func)
        tail = name.rsplit(".", 1)[-1]
        direct = name in imported_aliases or tail in EXECUTOR_TYPES
        qualified = any(
            name.startswith(f"{alias}.") and tail in EXECUTOR_TYPES
            for alias in module_aliases
        )
        if direct or qualified:
            findings.add((node.lineno, f"constructs {name or tail}"))
    return findings


def executor_ownership_violations(path: Path, tree: ast.AST) -> list[tuple[int, str]]:
    """Return direct executor ownership by an unreviewed module."""

    if path.relative_to(ROOT) in REVIEWED_EXECUTOR_OWNERS:
        return []
    imported_aliases, module_aliases, findings = _executor_import_ownership(tree)
    findings.update(_executor_construction_findings(tree, imported_aliases, module_aliases))
    return sorted(findings)


def main() -> int:
    files = source_files()
    print(f"PYTHON_FILES={len(files)}")

    for label, pattern in PATTERNS.items():
        print(f"=== {label} ===")
        count = 0
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                print(f"READ_ERROR {path.relative_to(ROOT)}: {exc}")
                continue
            for lineno, line in enumerate(lines, 1):
                if pattern.search(line):
                    count += 1
                    print(f"{path.relative_to(ROOT)}:{lineno}:{line.strip()}")
        print(f"COUNT={count}")

    print("=== syntax ===")
    bad = 0
    executor_violations = 0
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            bad += 1
            lineno = int(getattr(exc, "lineno", 0) or 0)
            print(f"SYNTAX_ERROR {path.relative_to(ROOT)}:{lineno}:{exc}")
            continue
        for lineno, detail in executor_ownership_violations(path, tree):
            executor_violations += 1
            print(
                "UNREVIEWED_EXECUTOR_OWNERSHIP "
                f"{path.relative_to(ROOT)}:{lineno}:{detail}; "
                "use minecraft_mod_ai.deadline_executor or explicitly justify scheduler ownership"
            )

    print(f"SYNTAX_ERRORS={bad}")
    print(f"UNREVIEWED_EXECUTOR_OWNERSHIP_ERRORS={executor_violations}")
    return 1 if bad or executor_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
