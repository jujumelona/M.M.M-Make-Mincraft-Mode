from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "minecraft_mod_ai"
_METADATA_ATTRS = {"__wrapped__", "__name__", "__qualname__", "__doc__", "__module__"}


def _root_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def _target_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _metadata_attribute(name: str) -> bool:
    return name.startswith("_mmm_") or name in _METADATA_ATTRS


class MutationVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[str] = []
        self.findings: list[dict[str, Any]] = []
        self.local_names: list[set[str]] = [set()]

    def _record(self, node: ast.AST, *, kind: str, target: str, metadata: bool = False) -> None:
        self.findings.append(
            {
                "path": self.path.relative_to(ROOT).as_posix(),
                "line": int(getattr(node, "lineno", 0)),
                "scope": ".".join(self.scope) or "<module>",
                "kind": kind,
                "target": target,
                "metadata_only": bool(metadata),
            }
        )

    def _remember_target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.local_names[-1].add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                self._remember_target(item)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.scope.append(node.name)
        names = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        if node.args.vararg:
            names.add(node.args.vararg.arg)
        if node.args.kwarg:
            names.add(node.args.kwarg.arg)
        self.local_names.append(names)
        for statement in node.body:
            self.visit(statement)
        self.local_names.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope.append(node.name)
        self.local_names.append(set())
        for statement in node.body:
            self.visit(statement)
        self.local_names.pop()
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            self._inspect_assignment_target(node, target)
            self._remember_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._inspect_assignment_target(node, node.target)
        self._remember_target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._inspect_assignment_target(node, node.target)
        self.visit(node.value)

    def _inspect_assignment_target(self, node: ast.AST, target: ast.AST) -> None:
        if isinstance(target, ast.Attribute):
            root = _root_name(target)
            # self/cls mutation is object-internal state, not runtime patching.
            if root in {"self", "cls"}:
                return
            self._record(
                node,
                kind="attribute_assignment",
                target=_target_text(target),
                metadata=_metadata_attribute(target.attr),
            )
            return
        if isinstance(target, ast.Subscript):
            text = _target_text(target)
            root = _root_name(target)
            # Namespace/sys.modules retargeting is behavior mutation. Ordinary local
            # dict/list updates are intentionally not reported.
            if root in {"namespace", "modules", "sys"} or "sys.modules" in text or "vars(" in text:
                self._record(node, kind="namespace_assignment", target=text)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id == "setattr" and len(node.args) >= 2:
            attr = node.args[1]
            attr_name = attr.value if isinstance(attr, ast.Constant) and isinstance(attr.value, str) else ""
            target = _target_text(node.args[0])
            if _root_name(node.args[0]) not in {"self", "cls"}:
                self._record(
                    node,
                    kind="setattr",
                    target=f"setattr({target}, {attr_name or '?'})",
                    metadata=bool(attr_name and _metadata_attribute(attr_name)),
                )
        self.generic_visit(node)


def audit() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            findings.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "line": int(getattr(exc, "lineno", 0) or 0),
                    "scope": "<parse>",
                    "kind": "parse_error",
                    "target": str(exc),
                    "metadata_only": False,
                }
            )
            continue
        visitor = MutationVisitor(path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-behavioral", action="store_true")
    args = parser.parse_args()
    findings = audit()
    behavioral = [item for item in findings if not item["metadata_only"]]
    payload = {
        "schema_version": "mmm/runtime-mutation-audit-v1",
        "behavioral_count": len(behavioral),
        "metadata_count": len(findings) - len(behavioral),
        "behavioral": behavioral,
        "metadata": [item for item in findings if item["metadata_only"]],
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 1 if args.fail_on_behavioral and behavioral else 0


if __name__ == "__main__":
    raise SystemExit(main())
