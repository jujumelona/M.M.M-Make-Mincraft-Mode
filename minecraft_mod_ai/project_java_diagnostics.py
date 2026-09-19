from __future__ import annotations

"""Target-aware Java/JDT diagnostics helpers.

Project Java selection is resolved at the verifier boundary, after project files exist.
This module contains the reusable implementation only; it never mutates
``ProductionToolService`` at import or bootstrap time.
"""

import os
import re
from pathlib import Path
from typing import Any

from .jdtls_bootstrap import ensure_jdtls

_MAX_BUILD_FILE_BYTES = 1024 * 1024
_PROJECT_BUILD_FILES = (
    "build.gradle",
    "build.gradle.kts",
    "gradle.properties",
)
_PROJECT_JAVA_PATTERNS = (
    re.compile(r"JavaLanguageVersion\s*\.\s*of\s*\(\s*(?P<major>\d+)\s*\)"),
    re.compile(r"JavaVersion\s*\.\s*VERSION_(?P<major>\d+)\b"),
    re.compile(
        r"\b(?:sourceCompatibility|targetCompatibility)\b\s*(?:=|:)\s*"
        r"(?:JavaVersion\s*\.\s*VERSION_)?[\"']?(?P<major>\d+)[\"']?\b"
    ),
    re.compile(
        r"\b(?:options\s*\.\s*)?release(?:\s*\.\s*set)?\s*"
        r"(?:\(|=)\s*(?P<major>\d+)\s*\)?"
    ),
    re.compile(
        r"(?im)^\s*(?:java_version|javaVersion|java_toolchain_version)\s*=\s*"
        r"(?P<major>\d+)\s*$"
    ),
)
_RELEASE_NOT_FOUND = re.compile(
    r"\brelease\s+(?P<major>\d+)\s+is\s+not\s+found\s+in\s+the\s+system\b",
    re.IGNORECASE,
)


class ProjectJavaResolutionError(RuntimeError):
    pass


def _read_build_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    size = path.stat().st_size
    if size > _MAX_BUILD_FILE_BYTES:
        raise ProjectJavaResolutionError(
            f"Project Java configuration file exceeds {_MAX_BUILD_FILE_BYTES} bytes: {path}"
        )
    return path.read_text(encoding="utf-8", errors="strict")


def _infer_project_java_major(project_root: str | Path) -> int | None:
    root = Path(project_root).expanduser().resolve()
    majors: set[int] = set()
    evidence: list[str] = []
    for relative in _PROJECT_BUILD_FILES:
        text = _read_build_file(root / relative)
        if not text:
            continue
        for pattern in _PROJECT_JAVA_PATTERNS:
            for match in pattern.finditer(text):
                major = int(match.group("major"))
                if major < 1:
                    continue
                majors.add(major)
                evidence.append(f"{relative}:{major}")
    if len(majors) > 1:
        raise ProjectJavaResolutionError(
            "Conflicting project Java requirements were found; refusing to guess: "
            + ", ".join(sorted(evidence))
        )
    return next(iter(majors)) if majors else None


def _toolchain_token() -> tuple[str, str]:
    return (
        os.environ.get("MMM_JAVA_VERSION", "").strip(),
        os.environ.get("MMM_PROJECT_JAVA_HOME", "").strip(),
    )


def _missing_release_major(exc: BaseException) -> int | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(16):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        match = _RELEASE_NOT_FOUND.search(str(current))
        if match is not None:
            return int(match.group("major"))
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return None


def _reset_cached_java_service(service: Any) -> None:
    java = service.__dict__.get("java")
    if java is None:
        return
    close = getattr(java, "close", None)
    if callable(close):
        close()
    service.__dict__.pop("java", None)


def _apply_project_java(service: Any, major: int) -> bool:
    before = _toolchain_token()
    os.environ["MMM_JAVA_VERSION"] = str(major)
    ensure_jdtls()
    changed = _toolchain_token() != before
    if changed:
        _reset_cached_java_service(service)
    return changed


def run_project_java_diagnostics(
    service: Any,
    project_root: str,
    relative_files: list[str] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    root = service._existing_dir(project_root)
    inferred = _infer_project_java_major(root)
    if inferred is not None:
        _apply_project_java(service, inferred)

    try:
        return service.java.diagnostics(
            root,
            relative_files=relative_files,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        required = _missing_release_major(exc)
        if required is None or not _apply_project_java(service, required):
            raise
        return service.java.diagnostics(
            root,
            relative_files=relative_files,
            timeout_seconds=timeout_seconds,
        )


__all__ = [
    "ProjectJavaResolutionError",
    "_infer_project_java_major",
    "_missing_release_major",
    "run_project_java_diagnostics",
]
