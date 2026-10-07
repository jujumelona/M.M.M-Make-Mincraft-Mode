from __future__ import annotations

"""Fast structural preflight for the MMM Python package.

This catches half-applied refactors (missing internal modules) and syntax errors
before planner/model or production work starts.  It does not import optional
third-party dependencies and therefore remains a lightweight host-only check.
"""

import ast
from pathlib import Path


class InternalPackagePreflightError(RuntimeError):
    pass


def _module_exists(package_root: Path, qualified: str) -> bool:
    prefix = package_root.name
    if qualified == prefix:
        return (package_root / "__init__.py").is_file()
    if not qualified.startswith(prefix + "."):
        return True
    relative = qualified.split(".")[1:]
    base = package_root.joinpath(*relative)
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _relative_target(
    package_root: Path,
    source: Path,
    *,
    level: int,
    module: str,
) -> str | None:
    if level < 1:
        return None
    relative_parent = source.relative_to(package_root).parent
    package_parts = [package_root.name, *relative_parent.parts]
    trim = level - 1
    if trim > len(package_parts) - 1:
        return None
    base = package_parts[: len(package_parts) - trim]
    module_parts = [part for part in module.split(".") if part]
    return ".".join([*base, *module_parts])


def validate_internal_package_integrity() -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    failures: list[str] = []
    file_count = 0
    internal_import_count = 0

    for source in sorted(package_root.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        file_count += 1
        try:
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(source))
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append(
                f"{source.relative_to(package_root)}: {type(exc).__name__}: {exc}"
            )
            continue

        for node in ast.walk(tree):
            target: str | None = None
            if isinstance(node, ast.ImportFrom):
                if node.level and node.module:
                    target = _relative_target(
                        package_root,
                        source,
                        level=node.level,
                        module=node.module,
                    )
                    if target is None:
                        failures.append(
                            f"{source.relative_to(package_root)}: "
                            f"invalid relative import level={node.level} "
                            f"module={node.module!r}"
                        )
                elif node.level and node.module is None:
                    base = _relative_target(
                        package_root,
                        source,
                        level=node.level,
                        module="",
                    )
                    if base is None:
                        failures.append(
                            f"{source.relative_to(package_root)}: "
                            f"invalid relative import level={node.level}"
                        )
                    else:
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            internal_import_count += 1
                            candidate = f"{base}.{alias.name}"
                            if not _module_exists(package_root, candidate):
                                failures.append(
                                    f"{source.relative_to(package_root)}: "
                                    f"missing internal module {candidate}"
                                )
                    continue
                elif node.module and (
                    node.module == package_root.name
                    or node.module.startswith(package_root.name + ".")
                ):
                    target = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = str(alias.name)
                    if name == package_root.name or name.startswith(
                        package_root.name + "."
                    ):
                        internal_import_count += 1
                        if not _module_exists(package_root, name):
                            failures.append(
                                f"{source.relative_to(package_root)}: "
                                f"missing internal module {name}"
                            )
                continue

            if target is not None:
                internal_import_count += 1
                if not _module_exists(package_root, target):
                    failures.append(
                        f"{source.relative_to(package_root)}: "
                        f"missing internal module {target}"
                    )

    if failures:
        raise InternalPackagePreflightError(
            "INTERNAL_PACKAGE_PREFLIGHT_FAILED: " + "; ".join(failures[:20])
        )
    return {
        "schema_version": "mmm/internal-package-preflight-v1",
        "status": "PASS",
        "python_file_count": file_count,
        "internal_import_count": internal_import_count,
    }


__all__ = [
    "InternalPackagePreflightError",
    "validate_internal_package_integrity",
]
