from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "minecraft_mod_ai"
WORKFLOWS = ROOT / ".github" / "workflows"

TOMBSTONED_OWNER_MODULES = (
    "minecraft_mod_ai/colab_mtp_server.py",
    "minecraft_mod_ai/execution_efficiency_contract.py",
    "minecraft_mod_ai/max_efficiency_runtime_contract.py",
    "minecraft_mod_ai/production_stream_resume_contract.py",
    "minecraft_mod_ai/scheduler_connection_reuse_contract.py",
)

_TRANSIENT_WORKFLOW_MARKERS = (
    "one-time",
    "one_time",
    "onetime",
    "-once",
    "_once",
)


def _module_exists(parts: list[str]) -> bool:
    if not parts or parts[0] != "minecraft_mod_ai":
        return True
    rel = parts[1:]
    if not rel:
        return (PKG / "__init__.py").is_file()
    file_path = PKG.joinpath(*rel).with_suffix(".py")
    package_path = PKG.joinpath(*rel, "__init__.py")
    return file_path.is_file() or package_path.is_file()


def _current_package(path: Path) -> list[str]:
    rel = path.relative_to(PKG)
    if rel.name == "__init__.py":
        return ["minecraft_mod_ai", *rel.parent.parts]
    return ["minecraft_mod_ai", *rel.parent.parts]


def _resolve_from(path: Path, node: ast.ImportFrom) -> list[str] | None:
    if node.level == 0:
        if node.module and node.module.startswith("minecraft_mod_ai"):
            return node.module.split(".")
        return None
    package = _current_package(path)
    up = node.level - 1
    if up > len(package) - 1:
        return ["<invalid-relative-depth>"]
    base = package[: len(package) - up]
    if node.module:
        base.extend(node.module.split("."))
    return base


def audit_internal_imports() -> list[str]:
    errors: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"SYNTAX {path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve_from(path, node)
            if target is None:
                continue
            if target == ["<invalid-relative-depth>"] or not _module_exists(target):
                errors.append(
                    f"MISSING_IMPORT {path.relative_to(ROOT)}:{node.lineno}: "
                    f"from {'.' * node.level}{node.module or ''} import ... -> "
                    f"{'.'.join(target)}"
                )
                continue
            if node.level and node.module is None:
                package = _current_package(path)
                up = node.level - 1
                base = package[: len(package) - up]
                for alias in node.names:
                    candidate = [*base, alias.name]
                    candidate_file = PKG.joinpath(*candidate[1:]).with_suffix(".py")
                    candidate_pkg = PKG.joinpath(*candidate[1:], "__init__.py")
                    if alias.name.islower() and "_" in alias.name:
                        if not candidate_file.is_file() and not candidate_pkg.is_file():
                            errors.append(
                                f"MISSING_SIBLING_IMPORT {path.relative_to(ROOT)}:{node.lineno}: "
                                f"from {'.' * node.level} import {alias.name}"
                            )
    return errors


def _getattr_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
        return None
    if len(node.args) < 2:
        return None
    name = node.args[1]
    return name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None


def _local_function_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Walk one function body without attributing nested functions to its owner."""

    nodes: list[ast.AST] = []
    stack: list[ast.AST] = list(reversed(function.body))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


def _nodes_call_named(nodes: list[ast.AST], name: str) -> bool:
    for item in nodes:
        if not isinstance(item, ast.Call):
            continue
        fn = item.func
        if isinstance(fn, ast.Name) and fn.id == name:
            return True
        if isinstance(fn, ast.Attribute) and fn.attr == name:
            return True
    return False


def audit_marker_controlled_unwraps() -> list[str]:
    """Reject inherited marker tests that can choose the wrong wrapped layer.

    Default ``functools.wraps`` copies function ``__dict__`` metadata, so direct
    ``getattr(current, '_mmm_*')`` cannot establish exact ownership. Simple
    installation/idempotence checks may still use inherited visibility, but any
    function that also *reads* ``__wrapped__`` must make ownership explicit through
    ``owns_contract_marker``. Nested wrapper definitions are audited independently.
    """

    errors: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes = _local_function_nodes(function)
            marker_lookup = False
            unwrap_lookup = False
            for node in nodes:
                name = _getattr_name(node)
                if isinstance(name, str) and name.startswith("_mmm_"):
                    marker_lookup = True
                if name == "__wrapped__":
                    unwrap_lookup = True
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "__wrapped__"
                    and isinstance(node.ctx, ast.Load)
                ):
                    unwrap_lookup = True
            if (
                marker_lookup
                and unwrap_lookup
                and not _nodes_call_named(nodes, "owns_contract_marker")
            ):
                errors.append(
                    f"UNSAFE_MARKER_CONTROLLED_UNWRAP {path.relative_to(ROOT)}:"
                    f"{function.lineno}: {function.name} must use owns_contract_marker"
                )
    return errors


def audit_bootstrap_owner_modules() -> list[str]:
    path = PKG / "runtime_bootstrap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    errors: list[str] = []
    installer_aliases: set[str] = set()
    called_installers: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            target = ["minecraft_mod_ai", *node.module.split(".")]
            if not _module_exists(target):
                errors.append(
                    f"BOOTSTRAP_MISSING_MODULE runtime_bootstrap.py:{node.lineno}: {node.module}"
                )
            for alias in node.names:
                if alias.name == "install":
                    installer_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id.startswith("install_"):
                called_installers[node.func.id] += 1
    missing_calls = sorted(name for name in installer_aliases if called_installers[name] == 0)
    if missing_calls:
        errors.append("BOOTSTRAP_IMPORTED_NOT_CALLED " + ",".join(missing_calls))
    duplicate_calls = sorted(
        (name, called_installers[name])
        for name in installer_aliases
        if called_installers[name] > 1
    )
    if duplicate_calls:
        errors.append(
            "BOOTSTRAP_INSTALLER_CALLED_MULTIPLE_TIMES "
            + ",".join(f"{name}={count}" for name, count in duplicate_calls)
        )
    return errors


def audit_tombstoned_owner_modules() -> list[str]:
    errors: list[str] = []
    for relative in TOMBSTONED_OWNER_MODULES:
        if (ROOT / relative).exists():
            errors.append(f"TOMBSTONED_OWNER_RESTORED {relative}")
    return errors


def _workflow_paths() -> list[Path]:
    if not WORKFLOWS.is_dir():
        return []
    return sorted({*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")})


def _top_level_mapping(node: yaml.Node) -> dict[str, yaml.Node]:
    if not isinstance(node, yaml.MappingNode):
        return {}
    values: dict[str, yaml.Node] = {}
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode):
            values[str(key_node.value)] = value_node
    return values


def audit_workflow_definitions() -> list[str]:
    errors: list[str] = []
    for path in _workflow_paths():
        relative = path.relative_to(ROOT)
        normalized_name = path.stem.casefold()
        if any(marker in normalized_name for marker in _TRANSIENT_WORKFLOW_MARKERS):
            errors.append(f"TRANSIENT_WORKFLOW_FORBIDDEN {relative}")
        try:
            node = yaml.compose(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = ""
            if mark is not None:
                location = f":{mark.line + 1}:{mark.column + 1}"
            errors.append(f"WORKFLOW_YAML_INVALID {relative}{location}: {exc}")
            continue
        if node is None or not isinstance(node, yaml.MappingNode):
            errors.append(f"WORKFLOW_ROOT_NOT_MAPPING {relative}")
            continue
        top = _top_level_mapping(node)
        if "on" not in top:
            errors.append(f"WORKFLOW_MISSING_ON {relative}")
        jobs = top.get("jobs")
        if not isinstance(jobs, yaml.MappingNode) or not jobs.value:
            errors.append(f"WORKFLOW_MISSING_JOBS {relative}")
    return errors


def main() -> int:
    errors = (
        audit_internal_imports()
        + audit_marker_controlled_unwraps()
        + audit_bootstrap_owner_modules()
        + audit_tombstoned_owner_modules()
        + audit_workflow_definitions()
    )
    if errors:
        print("STATIC DEBUG AUDIT FAILED")
        for error in errors:
            print(error)
        return 1
    py_count = sum(1 for _ in PKG.rglob("*.py"))
    workflow_count = len(_workflow_paths())
    print(
        "STATIC DEBUG AUDIT OK: "
        f"{py_count} package Python files and {workflow_count} workflows checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
