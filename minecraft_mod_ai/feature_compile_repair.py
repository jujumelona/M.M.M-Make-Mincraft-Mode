"""Compile-backed, bounded repair for one already-staged feature source.

The host compiles the *real* project. The model may propose at most two full
replacement versions of the task-owned file, grounded in compiler diagnostics.
This module does not claim runtime behavior or package a JAR.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .generation_target_compile import run_generation_target_compile
from .java_region_parser import _parse


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _owned_target(root: Path, relative: str) -> Path:
    if (not relative or not isinstance(relative, str) or
            not relative.startswith(("src/main/java/", "src/client/java/"))
            or not relative.endswith(".java") or "\\" in relative):
        raise ValueError("FEATURE_COMPILE_TARGET_INVALID")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("FEATURE_COMPILE_TARGET_OUTSIDE_PROJECT")
    if not result.is_file() or result.is_symlink():
        raise ValueError("FEATURE_COMPILE_TARGET_NOT_STAGED")
    return result


def _diagnostics(receipt: Mapping[str, Any], target: str) -> list[dict[str, Any]]:
    return [dict(row) for row in (receipt.get("diagnostics") or ())
            if isinstance(row, Mapping) and row.get("path") == target][:8]


def compile_and_repair_feature(
    project_root: str | Path,
    *,
    target_path: str,
    edit: Callable[[str, tuple[dict[str, Any], ...], int], str],
    max_repairs: int = 2,
    compile_target: Callable[..., Mapping[str, Any]] = run_generation_target_compile,
) -> dict[str, Any]:
    """Compile and fix only the currently owned Java file.

    `edit` receives the exact previous source and at most eight focused
    compiler errors; it must return a complete source replacement. A repeated
    source or repeated diagnostic set stops immediately. The caller owns the
    model prompt and is forbidden to change other files.
    """
    if type(max_repairs) is not int or not 0 <= max_repairs <= 2:
        raise ValueError("FEATURE_COMPILE_REPAIR_LIMIT_INVALID")
    root = Path(project_root).resolve()
    path = _owned_target(root, target_path)
    original = path.read_text(encoding="utf-8")
    if not original.strip():
        raise ValueError("FEATURE_COMPILE_EMPTY_SOURCE")
    seen_sources = {_digest(original)}
    seen_errors: set[tuple] = set()
    repairs = 0
    status = "FAILED"
    details: list[dict[str, Any]] = []
    try:
        while True:
            receipt = compile_target(root, target_path=target_path)
            raw = str(receipt.get("status", "")).upper()
            owned = _diagnostics(receipt, target_path)
            details.append({
                "status": raw, "target_path": target_path,
                "diagnostic_count": len(owned),
                "diagnostics": owned,
            })
            if raw == "PASS":
                status = "COMPILED"
                break
            if raw != "FAIL" or not owned:
                status = "COMPILER_UNAVAILABLE" if raw == "UNAVAILABLE" else "BLOCKED"
                break
            fingerprint = tuple(sorted((
                str(x.get("path", "")), str(x.get("line", "")),
                str(x.get("code", "")), str(x.get("message", "")),
            ) for x in owned))
            if fingerprint in seen_errors or repairs >= max_repairs:
                status = "REPAIR_EXHAUSTED"
                break
            seen_errors.add(fingerprint)
            current = path.read_text(encoding="utf-8")
            next_source = edit(current, tuple(owned), repairs + 1)
            repairs += 1
            if not isinstance(next_source, str) or not next_source.strip():
                status = "INVALID_REPAIR"
                break
            digest = _digest(next_source)
            if digest in seen_sources:
                status = "NO_PROGRESS"
                break
            seen_sources.add(digest)
            try:
                _parse(next_source)
            except Exception:
                status = "INVALID_REPAIR"
                break
            # Keep the target's package and primary class unchanged. The
            # actual compiler still establishes target API compatibility.
            from .construction_context import java_source_units
            try:
                before = java_source_units(original, path=target_path)
                after = java_source_units(next_source, path=target_path)
            except Exception:
                status = "INVALID_REPAIR"
                break
            before_owners = {
                (row.get("owner"), row.get("kind"))
                for row in before if row.get("kind") in {"class", "record", "enum", "interface"}
            }
            after_owners = {
                (row.get("owner"), row.get("kind"))
                for row in after if row.get("kind") in {"class", "record", "enum", "interface"}
            }
            if not before_owners or before_owners != after_owners:
                status = "INVALID_REPAIR"
                break
            path.write_text(next_source, encoding="utf-8")
    finally:
        if status != "COMPILED" and path.read_text(encoding="utf-8") != original:
            # Failed repair must never silently leave an unverified mutation.
            path.write_text(original, encoding="utf-8")
    return {
        "schema_version": "mmm/feature-compile-repair-v1",
        "status": status,
        "compile_passed": status == "COMPILED",
        "target_path": target_path,
        "repair_attempts": repairs,
        "original_sha256": _digest(original),
        "compiled_sha256": (_digest(path.read_text(encoding="utf-8"))
                            if status == "COMPILED" else ""),
        "history": details,
        "runtime_verified": False,
    }
