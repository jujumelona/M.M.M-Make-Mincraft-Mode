from __future__ import annotations

"""Repository-wide runtime-efficiency audit.

This audit is intentionally diagnostic rather than policy-by-number. It walks every
production Python source and reports concrete hot-path shapes that deserve review:
expensive calls performed directly in loops, repeated identical expensive calls,
per-call model/client construction, blocking waits, executor topology, and cache
ownership. It does not impose arbitrary feature/cardinality limits or fail a build
because a pattern merely exists.
"""

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
    ROOT / "minecraft_mod_ai",
    ROOT / "tools",
)
TOP_LEVEL_FILES = (
    ROOT / "download_resources.py",
)

_MODEL_TAILS = {
    "generate",
    "generate_turn",
    "generate_fixed_template_value",
    "run_record_template",
    "run_record_template_batch",
    "run_bounded_record_template",
    "run_single_record_template",
    "research_document_domain",
    "embed",
    "encode",
    "rerank",
    "generate_image",
    "render",
}
_NETWORK_TAILS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "request",
    "urlopen",
    "urlretrieve",
    "_json",
    "_text",
}
_PROCESS_TAILS = {
    "run",
    "Popen",
    "call",
    "check_call",
    "check_output",
    "system",
}
_WAIT_TAILS = {"sleep", "wait"}
_CONCURRENCY_TAILS = {
    "ThreadPoolExecutor",
    "ProcessPoolExecutor",
    "TaskGroup",
    "gather",
    "as_completed",
}
_CLIENT_TAILS = {
    "Client",
    "AsyncClient",
    "Session",
}
_MODEL_CONSTRUCTOR_SUFFIXES = ("Adapter", "Router")
_CACHE_MARKERS = ("cache", "memo", "pool")
_LOCK_TAILS = {"Lock", "RLock", "Semaphore", "BoundedSemaphore"}


def _source_paths() -> list[Path]:
    paths: set[Path] = set()
    for root in SCAN_ROOTS:
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    paths.update(path for path in TOP_LEVEL_FILES if path.is_file())
    return sorted(paths)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def _tail(call_name: str) -> str:
    return call_name.rsplit(".", 1)[-1]


def _call_kind(call_name: str) -> str | None:
    tail = _tail(call_name)
    lower = call_name.casefold()

    if tail in _MODEL_TAILS or (
        any(
            token in lower
            for token in (
                "model_router.generate",
                "router.generate",
                "llama",
                "inference",
            )
        )
        and tail in {"generate", "generate_turn", "complete"}
    ):
        return "model"

    if tail in _NETWORK_TAILS and (
        lower.startswith("httpx.")
        or lower.startswith("requests.")
        or "urllib" in lower
        or tail in {"urlopen", "urlretrieve", "_json", "_text"}
    ):
        return "network"

    if tail in _PROCESS_TAILS and (
        lower.startswith("subprocess.")
        or lower.startswith("os.system")
        or tail == "Popen"
    ):
        return "process"

    if tail in _WAIT_TAILS and (
        lower.startswith("time.")
        or lower.startswith("asyncio.")
        or lower.endswith(".wait")
    ):
        return "wait"

    return None


def _is_model_constructor(call_name: str) -> bool:
    tail = _tail(call_name)
    return tail == "ModelRouter" or tail.endswith(_MODEL_CONSTRUCTOR_SUFFIXES)


def _is_client_constructor(call_name: str) -> bool:
    tail = _tail(call_name)
    lower = call_name.casefold()
    return tail in _CLIENT_TAILS and (
        "httpx" in lower
        or "requests" in lower
        or "openai" in lower
        or "client" in lower
    )


def _is_executor_constructor(call_name: str) -> bool:
    return _tail(call_name) in {"ThreadPoolExecutor", "ProcessPoolExecutor"}


def _is_concurrency_call(call_name: str) -> bool:
    return _tail(call_name) in _CONCURRENCY_TAILS


def _iter_local_nodes(node: ast.AST) -> Iterable[ast.AST]:
    """Walk node but do not attribute nested callable/class bodies to an outer scope."""
    stack = [node]
    first = True
    while stack:
        current = stack.pop()
        if not first and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        first = False
        yield current
        children = list(ast.iter_child_nodes(current))
        stack.extend(reversed(children))


def _canonical_call(node: ast.Call) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _function_span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    end = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
    return max(1, end - int(node.lineno) + 1)


class EfficiencyVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.function_stack: list[str] = []
        self.loop_stack: list[int] = []
        self.findings: list[dict[str, Any]] = []
        self.inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.expensive_calls_by_function: dict[
            str, list[tuple[str, str, int, str]]
        ] = defaultdict(list)

    @property
    def function(self) -> str:
        return ".".join(self.function_stack) if self.function_stack else "<module>"

    def _record(
        self,
        category: str,
        node: ast.AST,
        *,
        detail: str,
        severity: str = "review",
        **extra: Any,
    ) -> None:
        self.findings.append(
            {
                "category": category,
                "severity": severity,
                "path": _relative(self.path),
                "line": int(getattr(node, "lineno", 0) or 0),
                "function": self.function,
                "detail": detail,
                **extra,
            }
        )

    def _inventory(self, category: str, node: ast.AST, **details: Any) -> None:
        self.inventory[category].append(
            {
                "path": _relative(self.path),
                "line": int(getattr(node, "lineno", 0) or 0),
                "function": self.function,
                **details,
            }
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        previous_loops = self.loop_stack
        self.loop_stack = []
        self._inventory(
            "function",
            node,
            name=node.name,
            span_lines=_function_span(node),
            async_function=isinstance(node, ast.AsyncFunctionDef),
        )
        for statement in node.body:
            self.visit(statement)
        self.loop_stack = previous_loops
        self.function_stack.pop()

    def visit_For(self, node: ast.For) -> Any:
        self.loop_stack.append(int(node.lineno))
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.loop_stack.append(int(node.lineno))
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_While(self, node: ast.While) -> Any:
        self.loop_stack.append(int(node.lineno))
        self.generic_visit(node)
        self.loop_stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = _name(node.func)
        kind = _call_kind(call_name)

        if kind:
            self._inventory(f"{kind}_call", node, call=call_name)
            if self.function != "<module>":
                self.expensive_calls_by_function[self.function].append(
                    (kind, call_name, int(node.lineno), _canonical_call(node))
                )
            if self.loop_stack:
                self._record(
                    f"serial_{kind}_call_in_loop",
                    node,
                    detail=(
                        f"{call_name} executes directly inside loop at line "
                        f"{self.loop_stack[-1]}"
                    ),
                    severity="high" if kind in {"model", "network", "process"} else "review",
                    loop_line=self.loop_stack[-1],
                    call=call_name,
                )

        if _is_model_constructor(call_name):
            self._inventory("model_constructor", node, call=call_name)
            if self.function != "<module>":
                self._record(
                    "model_or_adapter_constructed_in_function",
                    node,
                    detail=f"{call_name} is constructed during function execution",
                    severity="high" if self.loop_stack else "review",
                    loop_line=self.loop_stack[-1] if self.loop_stack else None,
                    call=call_name,
                )

        if _is_client_constructor(call_name):
            self._inventory("client_constructor", node, call=call_name)
            if self.function != "<module>":
                self._record(
                    "network_client_constructed_in_function",
                    node,
                    detail=f"{call_name} is constructed during function execution",
                    severity="high" if self.loop_stack else "review",
                    loop_line=self.loop_stack[-1] if self.loop_stack else None,
                    call=call_name,
                )

        if _is_executor_constructor(call_name):
            self._inventory(
                "executor_constructor",
                node,
                call=call_name,
                inside_loop=bool(self.loop_stack),
            )
            if self.loop_stack:
                self._record(
                    "executor_constructed_in_loop",
                    node,
                    detail=(
                        f"{call_name} is created inside loop at line "
                        f"{self.loop_stack[-1]}"
                    ),
                    severity="high",
                    loop_line=self.loop_stack[-1],
                    call=call_name,
                )

        if _is_concurrency_call(call_name):
            self._inventory("concurrency_call", node, call=call_name)

        if _tail(call_name) in _LOCK_TAILS:
            self._inventory("lock_constructor", node, call=call_name)

        self.generic_visit(node)

    def finalize(self) -> None:
        for function, calls in self.expensive_calls_by_function.items():
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
                        "severity": "high" if kind in {"model", "network", "process"} else "review",
                        "path": _relative(self.path),
                        "line": lines[0],
                        "function": function,
                        "detail": (
                            f"identical {kind} call {call_name} appears at lines "
                            + ", ".join(str(line) for line in lines)
                        ),
                        "call": call_name,
                        "lines": lines,
                    }
                )


def _cache_inventory(path: Path, tree: ast.Module) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    module_has_lock = any(
        isinstance(node, ast.Call) and _tail(_name(node.func)) in _LOCK_TAILS
        for top in tree.body
        for node in _iter_local_nodes(top)
    )

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                lower = target.id.casefold()
                if not any(marker in lower for marker in _CACHE_MARKERS):
                    continue
                results.append(
                    {
                        "path": _relative(path),
                        "line": int(node.lineno),
                        "scope": "module",
                        "name": target.id,
                        "module_has_lock": module_has_lock,
                    }
                )

    for fn in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__init__"
    ):
        for node in _iter_local_nodes(fn):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    continue
                lower = target.attr.casefold()
                if not any(marker in lower for marker in _CACHE_MARKERS):
                    continue
                results.append(
                    {
                        "path": _relative(path),
                        "line": int(node.lineno),
                        "scope": fn.name,
                        "name": f"self.{target.attr}",
                        "module_has_lock": module_has_lock,
                    }
                )
    return results


def _exact_duplicate_functions(
    function_records: list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, node in function_records:
        if not node.body:
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        canonical = ast.dump(body, annotate_fields=True, include_attributes=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        groups[digest].append(
            {
                "path": _relative(path),
                "line": int(node.lineno),
                "name": node.name,
                "span_lines": _function_span(node),
            }
        )

    duplicates: list[dict[str, Any]] = []
    for digest, rows in groups.items():
        unique_paths = {row["path"] for row in rows}
        if len(rows) < 2 or len(unique_paths) < 2:
            continue
        duplicates.append(
            {
                "body_sha256": digest,
                "copies": sorted(rows, key=lambda row: (row["path"], row["line"])),
            }
        )
    return sorted(
        duplicates,
        key=lambda group: (
            -sum(int(row["span_lines"]) for row in group["copies"]),
            group["body_sha256"],
        ),
    )


def audit() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    inventory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cache_inventory: list[dict[str, Any]] = []
    function_records: list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    parse_errors: list[dict[str, Any]] = []
    paths = _source_paths()

    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append(
                {
                    "path": _relative(path),
                    "line": int(getattr(exc, "lineno", 0) or 0),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        function_records.extend(
            (path, node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        visitor = EfficiencyVisitor(path)
        visitor.visit(tree)
        visitor.finalize()
        findings.extend(visitor.findings)
        for category, rows in visitor.inventory.items():
            inventory[category].extend(rows)
        cache_inventory.extend(_cache_inventory(path, tree))

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in findings:
        identity = (
            row["category"],
            row["path"],
            row["line"],
            row["function"],
            row.get("call"),
            row.get("loop_line"),
        )
        deduped.setdefault(identity, row)

    findings = sorted(
        deduped.values(),
        key=lambda row: (
            {"high": 0, "review": 1}.get(str(row["severity"]), 2),
            row["category"],
            row["path"],
            int(row["line"]),
        ),
    )

    category_counts = dict(sorted(Counter(row["category"] for row in findings).items()))
    inventory_counts = dict(
        sorted((category, len(rows)) for category, rows in inventory.items())
    )

    return {
        "schema_version": "mmm/runtime-efficiency-audit-v1",
        "scanned_python_files": len(paths),
        "parse_errors": parse_errors,
        "finding_count": len(findings),
        "finding_counts": category_counts,
        "findings": findings,
        "inventory_counts": inventory_counts,
        "inventory": dict(sorted(inventory.items())),
        "cache_inventory": sorted(
            cache_inventory, key=lambda row: (row["path"], row["line"], row["name"])
        ),
        "exact_duplicate_function_bodies": _exact_duplicate_functions(function_records),
    }


def _print_report(payload: dict[str, Any]) -> None:
    print(
        "RUNTIME EFFICIENCY AUDIT: "
        f"files={payload['scanned_python_files']} "
        f"findings={payload['finding_count']} "
        f"parse_errors={len(payload['parse_errors'])}"
    )
    for category, count in payload["finding_counts"].items():
        print(f"  {category}: {count}")
    for row in payload["findings"]:
        print(
            f"  [{row['severity']}] {row['category']} "
            f"{row['path']}:{row['line']} {row['function']}: {row['detail']}"
        )

    duplicates = payload["exact_duplicate_function_bodies"]
    print(f"  exact_duplicate_function_body_groups: {len(duplicates)}")
    for group in duplicates:
        copies = ", ".join(
            f"{row['path']}:{row['line']}:{row['name']}"
            for row in group["copies"]
        )
        print(f"    {copies}")


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
