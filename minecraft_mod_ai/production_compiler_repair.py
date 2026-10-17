"""One compile-backed repair of a direct-authored production Java file.

Only exact source-hash-matching .mmm/construction_authorship records authorize
changes. Unlike plan/AST admission, this uses a failing Gradle compiler log.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _owned_direct_sources(root: Path) -> dict[str, str]:
    record_root = root / ".mmm" / "construction_authorship"
    owned: dict[str, str] = {}
    if not record_root.is_dir():
        return owned
    for path in sorted(record_root.glob("*.json"))[:256]:
        if path.is_symlink() or path.stat().st_size > 32 * 1024:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (row.get("schema_version") != "mmm/construction-authorship-v1"
                or row.get("status") != "DIRECT_SOURCE_RENDERED_RUNTIME_UNVERIFIED"):
            continue
        rel = str(row.get("target_path") or "")
        if not rel.startswith(("src/main/java/", "src/client/java/")) or not rel.endswith(".java"):
            continue
        file = (root / rel).resolve()
        if not file.is_relative_to(root) or not file.is_file() or file.is_symlink():
            continue
        if hashlib.sha256(file.read_bytes()).hexdigest() != row.get("source_sha256"):
            continue
        owned[rel] = row["source_sha256"]
    return owned


def repair_failed_build_with_model(
    router: Any,
    project_root: str | Path,
    build: Mapping[str, Any],
    *,
    max_files: int = 1,
) -> dict[str, Any]:
    """Do not turn GameTest, EULA or cache failure into model retries."""
    root = Path(project_root).resolve()
    receipt: dict[str, Any] = {
        "schema_version": "mmm/production-compile-repair-v1",
        "status": "NOT_APPLICABLE",
        "repaired_files": [],
        "build_recheck_required": False,
        "runtime_verified": False,
    }
    if build.get("status") == "PASS":
        return receipt
    if max_files != 1:
        raise ValueError("PRODUCTION_COMPILE_REPAIR_LIMIT_INVALID")
    owned = _owned_direct_sources(root)
    errors = [
        row for row in build.get("diagnostics", ())
        if isinstance(row, Mapping)
        and row.get("severity") == 1
        and str(row.get("source") or "") == "javac"
        and str(row.get("path") or "") in owned
    ]
    if not errors:
        receipt["status"] = "NO_OWNED_JAVAC_ERROR"
        return receipt
    paths = list(dict.fromkeys(str(row["path"]) for row in errors))
    if len(paths) != 1:
        receipt["status"] = "MULTIPLE_OWNED_FILES"
        return receipt
    from .focused_method_compile_repair import repair_staged_java_with_model
    repair = repair_staged_java_with_model(router, root, target_path=paths[0])
    receipt["repair"] = repair
    if repair.get("status") == "COMPILED" and repair.get("compile_passed") is True:
        receipt["status"] = "COMPILER_REPAIRED"
        receipt["repaired_files"] = paths
        receipt["build_recheck_required"] = True
    else:
        receipt["status"] = "REPAIR_FAILED"
    return receipt
