from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
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
    """Find rebinding of imported/runtime-owned objects, not ordinary object state."""

    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.path = path
        self.scope: list[str] = []
        self.findings: list[dict[str, Any]] = []
        self.external_names: list[set[str]] = [set()]
        self.namespace_names: list[set[str]] = [set()]
        self.constants = self._string_constants(tree)

    @staticmethod
    def _string_constants(tree: ast.Module) -> dict[str, str]:
        values: dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value.value
        return values

    @property
    def external(self) -> set[str]:
        return self.external_names[-1]

    @property
    def namespaces(self) -> set[str]:
        return self.namespace_names[-1]

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

    def _resolved_attr_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id, "")
        return ""

    def _is_external_expr(self, node: ast.AST) -> bool:
        root = _root_name(node)
        if root in self.external:
            return True
        text = _target_text(node)
        if "sys.modules" in text:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "vars":
            return bool(node.args and self._is_external_expr(node.args[0]))
        return False

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.external.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        for alias in node.names:
            if alias.name != "*":
                self.external.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        inherited = set(self.external)
        parameters = {
            arg.arg
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        }
        if node.args.vararg:
            parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            parameters.add(node.args.kwarg.arg)
        inherited.update(
            name
            for name in parameters
            if name == "module" or name.endswith("_module") or name.endswith("_cls")
        )
        self.scope.append(node.name)
        self.external_names.append(inherited)
        self.namespace_names.append(set(self.namespaces))
        for statement in node.body:
            self.visit(statement)
        self.namespace_names.pop()
        self.external_names.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        # A class defined in this file is a canonical owner, not an external target.
        self.scope.append(node.name)
        self.external_names.append(set(self.external))
        self.namespace_names.append(set(self.namespaces))
        for statement in node.body:
            self.visit(statement)
        self.namespace_names.pop()
        self.external_names.pop()
        self.scope.pop()

    def visit_For(self, node: ast.For) -> Any:
        text = _target_text(node.iter)
        added: list[str] = []
        if "sys.modules" in text or "sys.modules.items" in text or "sys.modules.values" in text:
            targets = []
            if isinstance(node.target, ast.Name):
                targets = [node.target.id]
            elif isinstance(node.target, (ast.Tuple, ast.List)):
                targets = [item.id for item in node.target.elts if isinstance(item, ast.Name)]
            for name in targets:
                if name not in self.external:
                    self.external.add(name)
                    added.append(name)
        self.visit(node.iter)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        for name in added:
            self.external.discard(name)

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            self._inspect_assignment_target(node, target)
        self.visit(node.value)
        for target in node.targets:
            self._propagate_alias(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._inspect_assignment_target(node, node.target)
        if node.value is not None:
            self.visit(node.value)
            self._propagate_alias(node.target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._inspect_assignment_target(node, node.target)
        self.visit(node.value)

    def _propagate_alias(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        if self._is_external_expr(value):
            self.external.add(target.id)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "vars"
            and value.args
            and self._is_external_expr(value.args[0])
        ):
            self.namespaces.add(target.id)

    def _inspect_assignment_target(self, node: ast.AST, target: ast.AST) -> None:
        if isinstance(target, ast.Attribute):
            root = _root_name(target)
            if root not in self.external:
                return
            self._record(
                node,
                kind="external_attribute_rebind",
                target=_target_text(target),
                metadata=_metadata_attribute(target.attr),
            )
            return
        if isinstance(target, ast.Subscript):
            text = _target_text(target)
            root = _root_name(target)
            if root in self.namespaces or "sys.modules" in text:
                self._record(node, kind="loaded_namespace_rebind", target=text)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id == "setattr" and len(node.args) >= 2:
            target_node = node.args[0]
            if self._is_external_expr(target_node):
                attr_name = self._resolved_attr_name(node.args[1])
                self._record(
                    node,
                    kind="external_setattr",
                    target=f"setattr({_target_text(target_node)}, {attr_name or '?'})",
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
        visitor = MutationVisitor(path, tree)
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
    metadata = [item for item in findings if item["metadata_only"]]
    payload = {
        "schema_version": "mmm/runtime-mutation-audit-v2",
        "behavioral_count": len(behavioral),
        "metadata_count": len(metadata),
        "behavioral_by_path": dict(sorted(Counter(item["path"] for item in behavioral).items())),
        "behavioral": behavioral,
        "metadata": metadata,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 1 if args.fail_on_behavioral and behavioral else 0


if __name__ == "__main__":
    raise SystemExit(main())
