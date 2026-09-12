from __future__ import annotations

"""Repository-wide diagnostic audit for expensive runtime call topology.

The scanner follows module-local call wrappers before reporting loop hotspots. Pure
helpers such as ``_text`` or ``encode`` are therefore not mistaken for network/model
work merely because of their names. Findings are diagnostic: semantic retry loops and
ordered pipelines still require human review before any parallelization.
"""

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (ROOT / "minecraft_mod_ai", ROOT / "tools")
TOP_LEVEL_FILES = (ROOT / "download_resources.py",)

_MODEL_TAILS = frozenset(
    {
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
    }
)
_NETWORK_PREFIXES = ("httpx.", "requests.", "urllib.")
_NETWORK_TAILS = frozenset({"urlopen", "urlretrieve"})
_PROCESS_PREFIXES = ("subprocess.",)
_PROCESS_TAILS = frozenset({"Popen"})
_WAIT_PREFIXES = ("time.sleep", "asyncio.sleep")
_EXECUTOR_TAILS = frozenset({"ThreadPoolExecutor", "ProcessPoolExecutor"})
_CONCURRENCY_TAILS = frozenset(
    {"ThreadPoolExecutor", "ProcessPoolExecutor", "TaskGroup", "gather", "as_completed"}
)
_CLIENT_TAILS = frozenset({"Client", "AsyncClient", "Session"})
_LOCK_TAILS = frozenset({"Lock", "RLock", "Semaphore", "BoundedSemaphore"})
_CACHE_MARKERS = ("cache", "memo", "pool")


def _source_paths() -> list[Path]:
    values: set[Path] = set()
    for root in SCAN_ROOTS:
        if root.is_dir():
            values.update(root.rglob("*.py"))
    values.update(path for path in TOP_LEVEL_FILES if path.is_file())
    return sorted(values)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _tail(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def _direct_kind(call_name: str) -> str | None:
    tail = _tail(call_name)
    lower = call_name.casefold()
    if tail in _MODEL_TAILS:
        return "model"
    if tail in _NETWORK_TAILS or lower.startswith(_NETWORK_PREFIXES):
        return "network"
    if tail in _PROCESS_TAILS or lower.startswith(_PROCESS_PREFIXES):
        return "process"
    if lower.startswith(_WAIT_PREFIXES) or lower.endswith(".wait"):
        return "wait"
    return None


def _local_callable_name(call_name: str) -> str:
    if not call_name:
        return ""
    if call_name.startswith("self.") or call_name.startswith("cls."):
        return _tail(call_name)
    return call_name if "." not in call_name else ""


def _function_nodes(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _local_nodes(node: ast.AST):
    stack = list(reversed(list(ast.iter_child_nodes(node))))
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield current
        stack.extend(reversed(list(ast.iter_child_nodes(current))))


def _function_call_graph(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    names = {node.name for node in functions}
    direct: dict[str, set[str]] = defaultdict(set)
    edges: dict[str, set[str]] = defaultdict(set)
    for function in functions:
        for node in _local_nodes(function):
            if not isinstance(node, ast.Call):
                continue
            call_name = _name(node.func)
            kind = _direct_kind(call_name)
            if kind:
                direct[function.name].add(kind)
            local = _local_callable_name(call_name)
            if local in names and local != function.name:
                edges[function.name].add(local)

    inherited = {name: set(kinds) for name, kinds in direct.items()}
    changed = True
    while changed:
        changed = False
        for owner, callees in edges.items():
            combined = set(inherited.get(owner, set()))
            for callee in callees:
                combined.update(inherited.get(callee, set()))
            if combined != inherited.get(owner, set()):
                inherited[owner] = combined
                changed = True
    return inherited, edges


def _classify_call(call_name: str, wrapper_kinds: dict[str, set[str]]) -> set[str]:
    direct = _direct_kind(call_name)
    if direct:
        return {direct}
    local = _local_callable_name(call_name)
    return set(wrapper_kinds.get(local, set()))


def _canonical_call(node: ast.Call) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _span(node: ast.AST) -> int:
    return max(1, int(getattr(node, "end_lineno", node.lineno) or node.lineno) - int(node.lineno) + 1)


class RuntimeVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, wrapper_kinds: dict[str, set[str]]) -> None:
        self.path = path
        self.wrapper_kinds = wrapper_kinds
        self.functions: list[str] = []
        self.loops: list[int] = []
        self.findings: list[dict[str, Any]] = []
        self.inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.calls_by_function: dict[str, list[tuple[str, str, int, str]]] = defaultdict(list)

    @property
    def function(self) -> str:
        return ".".join(self.functions) if self.functions else "<module>"

    def _inventory(self, category: str, node: ast.AST, **extra: Any) -> None:
        self.inventory[category].append(
            {
                "path": _relative(self.path),
                "line": int(getattr(node, "lineno", 0) or 0),
                "function": self.function,
                **extra,
            }
        )

    def _finding(self, category: str, node: ast.AST, detail: str, **extra: Any) -> None:
        self.findings.append(
            {
                "category": category,
                "severity": "review",
                "path": _relative(self.path),
                "line": int(getattr(node, "lineno", 0) or 0),
                "function": self.function,
                "detail": detail,
                **extra,
            }
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        old_loops = self.loops
        self.loops = []
        self._inventory("function", node, name=node.name, span_lines=_span(node))
        for statement in node.body:
            self.visit(statement)
        self.loops = old_loops
        self.functions.pop()

    def visit_For(self, node: ast.For) -> Any:
        self.loops.append(int(node.lineno))
        self.generic_visit(node)
        self.loops.pop()

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> Any:
        self.loops.append(int(node.lineno))
        self.generic_visit(node)
        self.loops.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = _name(node.func)
        kinds = _classify_call(call_name, self.wrapper_kinds)
        for kind in sorted(kinds):
            self._inventory(f"{kind}_call", node, call=call_name)
            if self.function != "<module>":
                self.calls_by_function[self.function].append(
                    (kind, call_name, int(node.lineno), _canonical_call(node))
                )
            if self.loops:
                self._finding(
                    f"serial_{kind}_call_in_loop",
                    node,
                    f"{call_name} executes in loop at line {self.loops[-1]}",
                    call=call_name,
                    loop_line=self.loops[-1],
                )

        tail = _tail(call_name)
        if tail == "ModelRouter" or tail.endswith("Adapter"):
            self._inventory("model_constructor", node, call=call_name)
            if self.function != "<module>":
                self._finding(
                    "model_or_adapter_constructed_in_function",
                    node,
                    f"{call_name} is constructed during function execution",
                    call=call_name,
                )
        if tail in _CLIENT_TAILS and any(
            token in call_name.casefold() for token in ("httpx", "requests", "openai", "client")
        ):
            self._inventory("client_constructor", node, call=call_name)
            if self.function != "<module>":
                self._finding(
                    "network_client_constructed_in_function",
                    node,
                    f"{call_name} is constructed during function execution",
                    call=call_name,
                )
        if tail in _EXECUTOR_TAILS:
            self._inventory("executor_constructor", node, call=call_name, inside_loop=bool(self.loops))
            if self.loops:
                self._finding(
                    "executor_constructed_in_loop",
                    node,
                    f"{call_name} is created in loop at line {self.loops[-1]}",
                    call=call_name,
                    loop_line=self.loops[-1],
                )
        if tail in _CONCURRENCY_TAILS:
            self._inventory("concurrency_call", node, call=call_name)
        if tail in _LOCK_TAILS:
            self._inventory("lock_constructor", node, call=call_name)
        self.generic_visit(node)

    def finalize(self) -> None:
        for function, calls in self.calls_by_function.items():
            grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
            for kind, call_name, line, canonical in calls:
                grouped[(kind, canonical)].append((line, call_name))
            for (kind, _canonical), occurrences in grouped.items():
                lines = sorted({line for line, _ in occurrences})
                if len(lines) < 2:
                    continue
                call_name = occurrences[0][1]
                self.findings.append(
                    {
                        "category": "repeated_identical_expensive_call",
                        "severity": "review",
                        "path": _relative(self.path),
                        "line": lines[0],
                        "function": function,
                        "detail": f"identical {kind} call {call_name} appears at lines {lines}",
                        "call": call_name,
                        "lines": lines,
                    }
                )


def _cache_inventory(path: Path, tree: ast.Module) -> list[dict[str, Any]]:
    module_has_lock = any(
        isinstance(node, ast.Call) and _tail(_name(node.func)) in _LOCK_TAILS
        for node in ast.walk(tree)
    )
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                name = target.id
                scope = "module_or_local"
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                name = f"self.{target.attr}"
                scope = "instance"
            else:
                continue
            if any(marker in name.casefold() for marker in _CACHE_MARKERS):
                rows.append(
                    {
                        "path": _relative(path),
                        "line": int(node.lineno),
                        "scope": scope,
                        "name": name,
                        "module_has_lock": module_has_lock,
                    }
                )
    return rows


def _duplicates(function_records):
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, node in function_records:
        body = ast.Module(body=node.body, type_ignores=[])
        canonical = ast.dump(body, annotate_fields=True, include_attributes=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        groups[digest].append(
            {"path": _relative(path), "line": int(node.lineno), "name": node.name, "span_lines": _span(node)}
        )
    return [
        {"body_sha256": digest, "copies": sorted(rows, key=lambda row: (row["path"], row["line"]))}
        for digest, rows in groups.items()
        if len(rows) > 1 and len({row["path"] for row in rows}) > 1
    ]


def audit() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    caches: list[dict[str, Any]] = []
    functions_all = []
    parse_errors: list[dict[str, Any]] = []
    paths = _source_paths()

    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append(
                {"path": _relative(path), "line": int(getattr(exc, "lineno", 0) or 0), "error": str(exc)}
            )
            continue
        functions = _function_nodes(tree)
        wrapper_kinds, _edges = _function_call_graph(functions)
        functions_all.extend((path, node) for node in functions)
        visitor = RuntimeVisitor(path, wrapper_kinds)
        visitor.visit(tree)
        visitor.finalize()
        findings.extend(visitor.findings)
        for category, rows in visitor.inventory.items():
            inventory[category].extend(rows)
        caches.extend(_cache_inventory(path, tree))

    deduped = {}
    for row in findings:
        identity = (
            row["category"], row["path"], row["line"], row["function"], row.get("call"), row.get("loop_line")
        )
        deduped.setdefault(identity, row)
    findings = sorted(
        deduped.values(), key=lambda row: (row["category"], row["path"], row["line"])
    )
    return {
        "schema_version": "mmm/runtime-efficiency-audit-v2-callgraph",
        "scanned_python_files": len(paths),
        "parse_errors": parse_errors,
        "finding_count": len(findings),
        "finding_counts": dict(sorted(Counter(row["category"] for row in findings).items())),
        "findings": findings,
        "inventory_counts": dict(sorted((key, len(value)) for key, value in inventory.items())),
        "inventory": dict(sorted(inventory.items())),
        "cache_inventory": sorted(caches, key=lambda row: (row["path"], row["line"], row["name"])),
        "exact_duplicate_function_bodies": sorted(
            _duplicates(functions_all),
            key=lambda group: -sum(int(row["span_lines"]) for row in group["copies"]),
        ),
    }


def _print_report(payload: dict[str, Any]) -> None:
    print(
        "RUNTIME EFFICIENCY AUDIT: "
        f"files={payload['scanned_python_files']} findings={payload['finding_count']} "
        f"parse_errors={len(payload['parse_errors'])}"
    )
    for category, count in payload["finding_counts"].items():
        print(f"  {category}: {count}")
    for row in payload["findings"]:
        print(
            f"  [{row['severity']}] {row['category']} {row['path']}:{row['line']} "
            f"{row['function']}: {row['detail']}"
        )
    duplicates = payload["exact_duplicate_function_bodies"]
    print(f"  exact_duplicate_function_body_groups: {len(duplicates)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit()
    _print_report(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 1 if payload["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
