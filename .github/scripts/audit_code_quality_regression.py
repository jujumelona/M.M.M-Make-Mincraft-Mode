from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def _imports(path: str, tree: ast.AST, modules: set[str]) -> set[str]:
    owner = _module_name(path)
    result: set[str] = set()
    owner_package = owner.rsplit(".", 1)[0] if "." in owner else owner
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidates = [alias.name]
                parts = alias.name.split(".")
                candidates.extend(
                    ".".join(parts[:index]) for index in range(len(parts) - 1, 0, -1)
                )
                result.update(
                    candidate for candidate in candidates if candidate in modules and candidate != owner
                )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                pkg = owner_package.split(".")
                keep = max(0, len(pkg) - node.level + 1)
                prefix = ".".join(pkg[:keep])
                base = f"{prefix}.{base}".strip(".")
            for alias in node.names:
                candidate = f"{base}.{alias.name}".strip(".")
                if candidate in modules and candidate != owner:
                    result.add(candidate)
                elif base in modules and base != owner:
                    result.add(base)
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


def compare_snapshots(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    base_functions = base.get("functions", {})
    head_functions = head.get("functions", {})
    for key, metrics in sorted(head_functions.items()):
        previous = base_functions.get(key)
        if previous is None:
            exceeded = {
                "complexity": metrics["complexity"] > MAX_NEW_COMPLEXITY,
                "span": metrics["span"] > MAX_NEW_FUNCTION_LINES,
                "parameters": metrics["parameters"] > MAX_NEW_PARAMETERS,
            }
            if any(exceeded.values()):
                violations.append(
                    {
                        "category": "new_function_hard_ceiling",
                        "subject": key,
                        "metrics": metrics,
                        "exceeded": exceeded,
                    }
                )
        elif metrics["complexity"] > previous["complexity"]:
            violations.append(
                {
                    "category": "complexity_regression",
                    "subject": key,
                    "before": previous["complexity"],
                    "after": metrics["complexity"],
                }
            )
        if (
            previous is not None
            and metrics["span"] > previous["span"]
            and metrics["span"] > MAX_NEW_FUNCTION_LINES
        ):
            violations.append(
                {
                    "category": "function_size_regression",
                    "subject": key,
                    "before": previous["span"],
                    "after": metrics["span"],
                }
            )
        if (
            previous is not None
            and metrics["parameters"] > previous["parameters"]
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

    for path, metrics in sorted(head.get("files", {}).items()):
        previous = base.get("files", {}).get(path)
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

    base_cycles = {tuple(row) for row in base.get("import_cycles", [])}
    for cycle in {tuple(row) for row in head.get("import_cycles", [])} - base_cycles:
        violations.append({"category": "new_import_cycle", "subject": list(cycle)})

    for pair in introduced_duplicate_pairs(
        head.get("duplicate_function_groups", []),
        base.get("duplicate_function_groups", []),
    ):
        violations.append(
            {"category": "new_duplicate_function_body", "subject": list(pair)}
        )

    base_serial = {tuple(row) for row in base.get("serial_expensive_loops", [])}
    for row in {tuple(row) for row in head.get("serial_expensive_loops", [])} - base_serial:
        violations.append({"category": "new_serial_expensive_loop", "subject": list(row)})

    for row in head.get("parse_errors", []):
        violations.append({"category": "python_parse_error", "subject": row})
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


def _ci_gate_dependencies(text: str) -> set[str]:
    lines = text.splitlines()
    gate_index: int | None = None
    gate_indent = -1
    for index, line in enumerate(lines):
        if line.strip() == "ci-gate:":
            gate_index = index
            gate_indent = len(line) - len(line.lstrip())
            break
    if gate_index is None:
        return set()

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
        payload = stripped.partition(":")[2].strip()
        if payload.startswith("[") and payload.endswith("]"):
            return {
                item.strip().strip("'\"")
                for item in payload[1:-1].split(",")
                if item.strip()
            }
        if payload:
            return {payload.strip().strip("'\"")}

        dependencies: set[str] = set()
        needs_indent = indent
        for dependency_line in lines[index + 1 :]:
            dependency = dependency_line.strip()
            if not dependency or dependency.startswith("#"):
                continue
            dependency_indent = len(dependency_line) - len(dependency_line.lstrip())
            if dependency_indent <= needs_indent:
                break
            if dependency.startswith("-"):
                value = dependency[1:].strip().strip("'\"")
                if value:
                    dependencies.add(value)
        return dependencies
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
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    values: dict[str, str] = {}
    for path in listed.splitlines():
        if not _is_source_path(path):
            continue
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=root,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            values[path] = result.stdout
    return values


def run_audit(root: Path = ROOT, base_ref: str = "HEAD^") -> dict[str, Any]:
    head = analyze_sources(_working_sources(root))
    try:
        base = analyze_sources(_git_sources(root, base_ref))
        base_available = True
    except (OSError, subprocess.CalledProcessError):
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
