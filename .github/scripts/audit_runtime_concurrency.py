from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (ROOT / "minecraft_mod_ai",)
CANONICAL_EXECUTOR_OWNERS = {
    Path("minecraft_mod_ai/deadline_executor.py"),
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


def executor_ownership_violations(path: Path, tree: ast.AST) -> list[tuple[int, str]]:
    """Return direct executor ownership outside the single scheduler module.

    The canonical scheduler is the only module allowed to construct concurrent-futures
    pools. Planning, research, repair, and asset callers must use its bounded API so
    deadlines, cancellation, and shutdown cannot diverge between call paths.
    """

    relative = path.relative_to(ROOT)
    if relative in CANONICAL_EXECUTOR_OWNERS:
        return []

    findings: set[tuple[int, str]] = set()
    imported_aliases: set[str] = set()
    module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "concurrent.futures":
            for alias in node.names:
                if alias.name in EXECUTOR_TYPES:
                    local_name = alias.asname or alias.name
                    imported_aliases.add(local_name)
                    findings.add((node.lineno, f"imports {alias.name} as {local_name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "concurrent.futures":
                    module_aliases.add(alias.asname or "concurrent.futures")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_name(node.func)
        tail = name.rsplit(".", 1)[-1]
        if name in imported_aliases or tail in EXECUTOR_TYPES:
            findings.add((node.lineno, f"constructs {name or tail}"))
            continue
        if any(name.startswith(f"{alias}.") and tail in EXECUTOR_TYPES for alias in module_aliases):
            findings.add((node.lineno, f"constructs {name}"))

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
                "DIRECT_EXECUTOR_OWNERSHIP "
                f"{path.relative_to(ROOT)}:{lineno}:{detail}; "
                "use minecraft_mod_ai.deadline_executor"
            )

    print(f"SYNTAX_ERRORS={bad}")
    print(f"DIRECT_EXECUTOR_OWNERSHIP_ERRORS={executor_violations}")
    return 1 if bad or executor_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
