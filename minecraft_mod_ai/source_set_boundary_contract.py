from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*;")
_IMPORT_RE = re.compile(
    r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$*][\w$*]*)*)\s*;"
)

# Loader/runtime packages which are intrinsically client-only even when the
# referenced class is supplied by a dependency rather than this project.
_CLIENT_ONLY_PREFIXES = (
    "net.minecraft.client",
    "com.mojang.blaze3d",
    "net.fabricmc.fabric.api.client",
)


@dataclass(frozen=True)
class JavaSourceUnit:
    relative_path: str
    source_set: str
    package: str
    simple_name: str
    qualified_name: str
    code: str


class SourceSetBoundaryError(ValueError):
    """A common/server Java source can reach a client-only dependency."""


def _strip_comments_and_literals(text: str) -> str:
    """Replace Java comments/string/char bodies while preserving token offsets."""

    output = list(text)
    state = "code"
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if state == "code":
            if char == "/" and next_char == "/":
                output[index] = output[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and next_char == "*":
                output[index] = output[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                output[index] = " "
                state = "string"
            elif char == "'":
                output[index] = " "
                state = "char"
            index += 1
            continue

        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                output[index] = " "
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                output[index] = output[index + 1] = " "
                state = "code"
                index += 2
            else:
                if char != "\n":
                    output[index] = " "
                index += 1
            continue

        # Java string/char literal. Escaped characters must not terminate it.
        if char == "\\":
            output[index] = " "
            if index + 1 < len(text):
                if text[index + 1] != "\n":
                    output[index + 1] = " "
                index += 2
            else:
                index += 1
            continue
        terminator = '"' if state == "string" else "'"
        if char == terminator:
            output[index] = " "
            state = "code"
        elif char != "\n":
            output[index] = " "
        index += 1

    return "".join(output)


def _source_set(relative: str) -> str | None:
    normalized = relative.replace("\\", "/")
    if normalized.startswith("src/client/java/"):
        return "client"
    if normalized.startswith("src/main/java/"):
        return "common"
    if normalized.startswith("src/server/java/"):
        return "server"
    return None


def _load_unit(root: Path, path: Path) -> JavaSourceUnit | None:
    relative = path.relative_to(root).as_posix()
    source_set = _source_set(relative)
    if source_set is None:
        return None
    text = path.read_text(encoding="utf-8")
    code = _strip_comments_and_literals(text)
    package_match = _PACKAGE_RE.search(code)
    package = package_match.group(1) if package_match else ""
    simple_name = path.stem
    qualified_name = f"{package}.{simple_name}" if package else simple_name
    return JavaSourceUnit(
        relative_path=relative,
        source_set=source_set,
        package=package,
        simple_name=simple_name,
        qualified_name=qualified_name,
        code=code,
    )


def _java_units(project_root: str | Path) -> tuple[JavaSourceUnit, ...]:
    root = Path(project_root).resolve(strict=True)
    units: list[JavaSourceUnit] = []
    for source_root in ("src/main/java", "src/server/java", "src/client/java"):
        directory = root / source_root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.java"), key=lambda item: item.as_posix()):
            if path.is_symlink() or not path.is_file():
                continue
            unit = _load_unit(root, path)
            if unit is not None:
                units.append(unit)
    return tuple(units)


def _is_external_client_reference(target: str) -> bool:
    normalized = target.removesuffix(".*")
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        for prefix in _CLIENT_ONLY_PREFIXES
    )


def source_set_boundary_errors(project_root: str | Path) -> tuple[str, ...]:
    """Return deterministic common/server -> client-only dependency violations.

    Every reachable path from common/server to a project client class has a first
    edge entering the client set, so rejecting every such direct edge also rejects
    transitive reachability. Imports, static imports, fully-qualified references,
    and same-package simple-name references are covered.
    """

    units = _java_units(project_root)
    client_units = tuple(unit for unit in units if unit.source_set == "client")
    protected = tuple(unit for unit in units if unit.source_set in {"common", "server"})
    client_by_fqn = {unit.qualified_name: unit for unit in client_units}
    client_packages: dict[str, set[str]] = {}
    for unit in client_units:
        client_packages.setdefault(unit.package, set()).add(unit.simple_name)

    findings: set[str] = set()
    for unit in protected:
        imports = tuple(match.group(1) for match in _IMPORT_RE.finditer(unit.code))
        for target in imports:
            if _is_external_client_reference(target):
                findings.add(
                    f"{unit.relative_path}: {unit.source_set} source imports client-only dependency {target}"
                )
                continue
            normalized = target.removesuffix(".*")
            if target.endswith(".*"):
                if any(fqn.startswith(normalized + ".") for fqn in client_by_fqn):
                    findings.add(
                        f"{unit.relative_path}: {unit.source_set} source imports client source package {target}"
                    )
                continue
            if normalized in client_by_fqn or any(
                normalized.startswith(fqn + ".") for fqn in client_by_fqn
            ):
                findings.add(
                    f"{unit.relative_path}: {unit.source_set} source imports client source {target}"
                )

        # Import-free fully-qualified references are still dependency edges.
        for fqn in client_by_fqn:
            if re.search(rf"(?<![\w$]){re.escape(fqn)}(?![\w$])", unit.code):
                findings.add(
                    f"{unit.relative_path}: {unit.source_set} source references client source {fqn}"
                )

        # Java permits same-package references without imports.
        for simple_name in client_packages.get(unit.package, set()):
            if re.search(rf"(?<![\w$]){re.escape(simple_name)}(?![\w$])", unit.code):
                findings.add(
                    f"{unit.relative_path}: {unit.source_set} source references same-package client source {simple_name}"
                )

    return tuple(sorted(findings))


def assert_server_safe_source_sets(project_root: str | Path) -> None:
    errors = source_set_boundary_errors(project_root)
    if errors:
        rendered = "\n".join(f"- {item}" for item in errors)
        raise SourceSetBoundaryError(
            "Java source-set boundary violation; common/server code must not reach client-only dependencies:\n"
            + rendered
        )
