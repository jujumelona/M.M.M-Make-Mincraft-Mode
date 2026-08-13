from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "minecraft_mod_ai"


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


def main() -> int:
    errors = audit_internal_imports() + audit_bootstrap_owner_modules()
    if errors:
        print("STATIC DEBUG AUDIT FAILED")
        for error in errors:
            print(error)
        return 1
    py_count = sum(1 for _ in PKG.rglob("*.py"))
    print(f"STATIC DEBUG AUDIT OK: {py_count} package Python files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
