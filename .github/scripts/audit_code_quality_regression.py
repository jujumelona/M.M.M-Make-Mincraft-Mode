from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_duplicate_pairs import introduced_duplicate_pairs

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PREFIXES = ("minecraft_mod_ai/", "tools/", ".github/scripts/")
TOP_LEVEL = {"download_resources.py"}
EXPENSIVE_TAILS = {
    "generate",
    "generate_turn",
    "generate_fixed_template_value",
    "run_record_template",
    "run_record_template_batch",
    "run_bounded_record_template",
    "run_single_record_template",
    "research_document_domain",
    "embed",
    "rerank",
    "generate_image",
    "urlopen",
    "urlretrieve",
    "Popen",
    "wait",
    "sleep",
}
MAX_NEW_COMPLEXITY = 15
MAX_NEW_FUNCTION_LINES = 120
MAX_NEW_PARAMETERS = 10
MAX_NEW_FILE_LINES = 1200


def _module_name(path: str) -> str:
    value = path[:-3] if path.endswith(".py") else path
    parts = value.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _complexity(node: ast.AST) -> int:
    score = 1
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Assert)):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += max(0, len(child.values) - 1)
        elif isinstance(child, ast.Try):
            score += len(child.handlers) + int(bool(child.orelse))
        elif isinstance(child, ast.Match):
            score += max(0, len(child.cases) - 1)
        elif isinstance(child, ast.comprehension):
            score += 1 + len(child.ifs)
    return score


def _function_span(node: ast.AST) -> int:
    start = int(getattr(node, "lineno", 0) or 0)
    end = int(getattr(node, "end_lineno", start) or start)
    return max(1, end - start + 1)


def _parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    return (
        len(args.posonlyargs)
        + len(args.args)
        + len(args.kwonlyargs)
        + int(args.vararg is not None)
        + int(args.kwarg is not None)
    )


def _canonical_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = ast.Module(body=node.body, type_ignores=[])
    return hashlib.sha256(
        ast.dump(body, annotate_fields=True, include_attributes=False).encode()
    ).hexdigest()


def _serial_expensive_calls(path: str, tree: ast.AST) -> set[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []
            self.loop_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self.functions.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def _visit_loop(self, node: ast.AST) -> None:
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        visit_For = _visit_loop
        visit_AsyncFor = _visit_loop
        visit_While = _visit_loop

        def visit_Call(self, node: ast.Call) -> Any:
            name = _call_name(node.func)
            tail = name.rsplit(".", 1)[-1]
            lower = name.casefold()
            expensive = tail in EXPENSIVE_TAILS or lower.startswith(
                ("requests.", "httpx.", "urllib.", "subprocess.")
            )
            if self.loop_depth and expensive:
                rows.add((path, ".".join(self.functions) or "<module>", name))
            self.generic_visit(node)

    Visitor().visit(tree)
    return rows


def _import_candidates(name: str) -> tuple[str, ...]:
    parts = name.split(".")
    parents = tuple(".".join(parts[:index]) for index in range(len(parts) - 1, 0, -1))
    return (name, *parents)


def _relative_import_base(owner_package: str, node: ast.ImportFrom) -> str:
    base = node.module or ""
    if not node.level:
        return base
    package_parts = owner_package.split(".")
    keep = max(0, len(package_parts) - node.level + 1)
    prefix = ".".join(package_parts[:keep])
    return f"{prefix}.{base}".strip(".")


def _import_from_targets(
    owner: str,
    owner_package: str,
    node: ast.ImportFrom,
    modules: set[str],
) -> set[str]:
    base = _relative_import_base(owner_package, node)
    targets: set[str] = set()
    for alias in node.names:
        candidate = f"{base}.{alias.name}".strip(".")
        if candidate in modules and candidate != owner:
            targets.add(candidate)
        elif base in modules and base != owner:
            targets.add(base)
    return targets


def _imports(path: str, tree: ast.AST, modules: set[str]) -> set[str]:
    owner = _module_name(path)
    owner_package = owner.rsplit(".", 1)[0] if "." in owner else owner
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = (
                candidate
                for alias in node.names
                for candidate in _import_candidates(alias.name)
            )
            result.update(
                candidate for candidate in candidates if candidate in modules and candidate != owner
            )
        elif isinstance(node, ast.ImportFrom):
            result.update(_import_from_targets(owner, owner_package, node, modules))
    return result


def _cycles(graph: dict[str, set[str]]) -> set[tuple[str, ...]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    cycles: set[tuple[str, ...]] = set()

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, set()):
            if target not in indices:
                strongconnect(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component: list[str] = []
            while stack:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1 or (
                len(component) == 1 and component[0] in graph.get(component[0], set())
            ):
                cycles.add(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            strongconnect(node)
    return cycles


def analyze_sources(sources: dict[str, str]) -> dict[str, Any]:
    trees: dict[str, ast.Module] = {}
    parse_errors: list[dict[str, Any]] = []
    functions: dict[str, dict[str, Any]] = {}
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    serial_calls: set[tuple[str, str, str]] = set()
    files: dict[str, dict[str, int]] = {}

    for path, source in sorted(sources.items()):
        files[path] = {"lines": len(source.splitlines())}
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            parse_errors.append(
                {"path": path, "line": int(exc.lineno or 0), "error": str(exc)}
            )
            continue
        trees[path] = tree
        serial_calls.update(_serial_expensive_calls(path, tree))
        parents: list[str] = []

        class FunctionVisitor(ast.NodeVisitor):
            def _visit(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                qualified = ".".join([*parents, node.name])
                key = f"{path}:{qualified}"
                functions[key] = {
                    "path": path,
                    "name": qualified,
                    "complexity": _complexity(node),
                    "span": _function_span(node),
                    "parameters": _parameter_count(node),
                }
                duplicate_groups[_canonical_body(node)].append(key)
                parents.append(node.name)
                for statement in node.body:
                    self.visit(statement)
                parents.pop()

            visit_FunctionDef = _visit
            visit_AsyncFunctionDef = _visit

        FunctionVisitor().visit(tree)

    modules = {_module_name(path) for path in trees}
    graph = {
        _module_name(path): _imports(path, tree, modules) for path, tree in trees.items()
    }
    duplicates = {
        tuple(sorted(items))
        for items in duplicate_groups.values()
        if len(items) > 1
        and len({item.split(":", 1)[0] for item in items}) > 1
    }
    return {
        "parse_errors": parse_errors,
        "files": files,
        "functions": functions,
        "import_cycles": sorted(_cycles(graph)),
        "duplicate_function_groups": sorted(duplicates),
        "serial_expensive_loops": sorted(serial_calls),
    }


def _new_function_violation(key: str, metrics: dict[str, Any]) -> dict[str, Any] | None:
    exceeded = {
        "complexity": metrics["complexity"] > MAX_NEW_COMPLEXITY,
        "span": metrics["span"] > MAX_NEW_FUNCTION_LINES,
        "parameters": metrics["parameters"] > MAX_NEW_PARAMETERS,
    }
    if not any(exceeded.values()):
        return None
    return {
        "category": "new_function_hard_ceiling",
        "subject": key,
        "metrics": metrics,
        "exceeded": exceeded,
    }


def _existing_function_violations(
    key: str,
    previous: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if metrics["complexity"] > previous["complexity"]:
        violations.append(
            {
                "category": "complexity_regression",
                "subject": key,
                "before": previous["complexity"],
                "after": metrics["complexity"],
            }
        )
    if metrics["span"] > previous["span"] and metrics["span"] > MAX_NEW_FUNCTION_LINES:
        violations.append(
            {
                "category": "function_size_regression",
                "subject": key,
                "before": previous["span"],
                "after": metrics["span"],
            }
        )
    if (
        metrics["parameters"] > previous["parameters"]
        and metrics["parameters"] > MAX_NEW_PARAMETERS
    ):
        violations.append(
            {
                "category": "parameter_count_regression",
                "subject": key,
                "before": previous["parameters"],
                "after": metrics["parameters"],
            }
        )
    return violations


def _function_regressions(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    base_functions = base.get("functions", {})
    for key, metrics in sorted(head.get("functions", {}).items()):
        previous = base_functions.get(key)
        if previous is None:
            violation = _new_function_violation(key, metrics)
            if violation is not None:
                violations.append(violation)
        else:
            violations.extend(_existing_function_violations(key, previous, metrics))
    return violations


def _file_regressions(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    base_files = base.get("files", {})
    for path, metrics in sorted(head.get("files", {}).items()):
        previous = base_files.get(path)
        if previous is None and metrics["lines"] > MAX_NEW_FILE_LINES:
            violations.append(
                {"category": "new_file_hard_ceiling", "subject": path, "lines": metrics["lines"]}
            )
        elif (
            previous is not None
            and metrics["lines"] > previous["lines"]
            and metrics["lines"] > MAX_NEW_FILE_LINES
        ):
            violations.append(
                {
                    "category": "file_size_regression",
                    "subject": path,
                    "before": previous["lines"],
                    "after": metrics["lines"],
                }
            )
    return violations


def _cycle_regressions(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    base_cycles = {tuple(row) for row in base.get("import_cycles", [])}
    head_cycles = {tuple(row) for row in head.get("import_cycles", [])}
    return [
        {"category": "new_import_cycle", "subject": list(cycle)}
        for cycle in sorted(head_cycles - base_cycles)
    ]


def _duplicate_regressions(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"category": "new_duplicate_function_body", "subject": list(pair)}
        for pair in introduced_duplicate_pairs(
            head.get("duplicate_function_groups", []),
            base.get("duplicate_function_groups", []),
        )
    ]


def _serial_loop_regressions(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    base_serial = {tuple(row) for row in base.get("serial_expensive_loops", [])}
    head_serial = {tuple(row) for row in head.get("serial_expensive_loops", [])}
    return [
        {"category": "new_serial_expensive_loop", "subject": list(row)}
        for row in sorted(head_serial - base_serial)
    ]


def compare_snapshots(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    violations = _function_regressions(base, head)
    violations.extend(_file_regressions(base, head))
    violations.extend(_cycle_regressions(base, head))
    violations.extend(_duplicate_regressions(base, head))
    violations.extend(_serial_loop_regressions(base, head))
    violations.extend(
        {"category": "python_parse_error", "subject": row}
        for row in head.get("parse_errors", [])
    )
    return violations


_REQUIRED_CI_TOKENS = (
    "audit_runtime_concurrency.py",
    "audit_runtime_efficiency.py",
    "audit_code_quality_regression.py",
    "verify_integrity_minecraft.py",
    "root_cause_audit_wrapper.py",
)
_REQUIRED_CI_GATE_DEPENDENCIES = frozenset(
    {"audit", "tests", "python313", "model-realistic-replay"}
)


def _find_ci_gate(lines: list[str]) -> tuple[int, int] | None:
    for index, line in enumerate(lines):
        if line.strip() == "ci-gate:":
            return index, len(line) - len(line.lstrip())
    return None


def _inline_dependencies(payload: str) -> set[str] | None:
    if payload.startswith("[") and payload.endswith("]"):
        return {
            item.strip().strip("'\"")
            for item in payload[1:-1].split(",")
            if item.strip()
        }
    if payload:
        return {payload.strip().strip("'\"")}
    return None


def _block_dependencies(lines: list[str], start: int, needs_indent: int) -> set[str]:
    dependencies: set[str] = set()
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= needs_indent:
            break
        if stripped.startswith("-"):
            value = stripped[1:].strip().strip("'\"")
            if value:
                dependencies.add(value)
    return dependencies


def _ci_gate_dependencies(text: str) -> set[str]:
    lines = text.splitlines()
    gate = _find_ci_gate(lines)
    if gate is None:
        return set()
    gate_index, gate_indent = gate
    for index in range(gate_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= gate_indent:
            break
        if not stripped.startswith("needs:"):
            continue
        inline = _inline_dependencies(stripped.partition(":")[2].strip())
        return inline if inline is not None else _block_dependencies(lines, index + 1, indent)
    return set()


def audit_main_ci_text(text: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if not ("pytest_diagnostics.py" in text and " tests" in text):
        violations.append({"category": "ci_missing_full_test_suite"})
    missing = [token for token in _REQUIRED_CI_TOKENS if token not in text]
    if missing:
        violations.append(
            {"category": "ci_missing_authoritative_audit", "missing": missing}
        )
    actual_dependencies = _ci_gate_dependencies(text)
    missing_dependencies = sorted(_REQUIRED_CI_GATE_DEPENDENCIES - actual_dependencies)
    if missing_dependencies:
        violations.append(
            {
                "category": "ci_gate_dependency_gap",
                "required": sorted(_REQUIRED_CI_GATE_DEPENDENCIES),
                "missing": missing_dependencies,
                "actual": sorted(actual_dependencies),
            }
        )
    return violations


def _is_source_path(path: str) -> bool:
    return path in TOP_LEVEL or (
        path.endswith(".py") and path.startswith(SOURCE_PREFIXES)
    )


def _working_sources(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for prefix in SOURCE_PREFIXES:
        directory = root / prefix
        if directory.is_dir():
            for path in directory.rglob("*.py"):
                values[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    for name in TOP_LEVEL:
        path = root / name
        if path.is_file():
            values[name] = path.read_text(encoding="utf-8")
    return values


def _git_sources(root: Path, ref: str) -> dict[str, str]:
    archived = subprocess.run(
        ["git", "archive", "--format=tar", ref],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    values: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archived), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _is_source_path(member.name):
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                values[member.name] = extracted.read().decode("utf-8", errors="replace")
    return values


def run_audit(root: Path = ROOT, base_ref: str = "HEAD^") -> dict[str, Any]:
    head = analyze_sources(_working_sources(root))
    try:
        base = analyze_sources(_git_sources(root, base_ref))
        base_available = True
    except (OSError, subprocess.CalledProcessError, tarfile.TarError):
        base = analyze_sources({})
        base_available = False
    violations = compare_snapshots(base, head)
    workflow = root / ".github" / "workflows" / "main-ci.yml"
    if workflow.is_file():
        violations.extend(audit_main_ci_text(workflow.read_text(encoding="utf-8")))
    else:
        violations.append({"category": "ci_missing_authoritative_workflow"})
    return {
        "schema": "mmm/ci-quality-regression",
        "base_ref": base_ref,
        "base_available": base_available,
        "head_summary": {
            "files": len(head["files"]),
            "functions": len(head["functions"]),
            "import_cycles": len(head["import_cycles"]),
            "duplicate_function_groups": len(head["duplicate_function_groups"]),
            "serial_expensive_loops": len(head["serial_expensive_loops"]),
        },
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", default="HEAD^")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_audit(base_ref=args.base_ref)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())