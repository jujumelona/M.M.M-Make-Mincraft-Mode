from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "minecraft_mod_ai"
WORKFLOWS = ROOT / ".github" / "workflows"

# These owners were intentionally retired after their responsibility moved elsewhere.
# Recreating them would reintroduce the exact split-ownership/stale-import failures that
# this audit exists to prevent.
TOMBSTONED_OWNER_MODULES = (
    "minecraft_mod_ai/colab_mtp_server.py",
    "minecraft_mod_ai/production_stream_resume_contract.py",
    "minecraft_mod_ai/qwen35_t4_single_stream_tuning.py",
)

# Direct-main development must not use self-deleting patch workflows. They race with
# normal commits and, when malformed, can create a failure on every subsequent push.
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
            # `from . import foo` imports a sibling module in runtime bootstrap-style
            # composition. Check aliases that clearly correspond to a live module.
            if node.level and node.module is None:
                package = _current_package(path)
                up = node.level - 1
                base = package[: len(package) - up]
                for alias in node.names:
                    candidate = [*base, alias.name]
                    candidate_file = PKG.joinpath(*candidate[1:]).with_suffix(".py")
                    candidate_pkg = PKG.joinpath(*candidate[1:], "__init__.py")
                    # Only flag module-shaped names. Uppercase/public attributes are not
                    # interpreted as sibling modules.
                    if alias.name.islower() and "_" in alias.name:
                        if not candidate_file.is_file() and not candidate_pkg.is_file():
                            errors.append(
                                f"MISSING_SIBLING_IMPORT {path.relative_to(ROOT)}:{node.lineno}: "
                                f"from {'.' * node.level} import {alias.name}"
                            )
    return errors


def audit_bootstrap_owner_modules() -> list[str]:
    path = PKG / "runtime_bootstrap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    errors: list[str] = []
    installer_aliases: set[str] = set()
    called_installers: set[str] = set()
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
                called_installers.add(node.func.id)
    missing_calls = sorted(installer_aliases - called_installers)
    if missing_calls:
        errors.append("BOOTSTRAP_IMPORTED_NOT_CALLED " + ",".join(missing_calls))
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
