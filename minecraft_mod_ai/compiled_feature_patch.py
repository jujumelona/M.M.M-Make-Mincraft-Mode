"""Replay a compiler-proven source correction after a hash-identical regen.

The source patch is bound to the original full Java bytes, the requested
file, and the project configuration. Never apply arbitrary diff hunks or stale
patches across Minecraft versions, loaders or changed feature signatures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .java_region_parser import _parse


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _project_hash(root: Path) -> str:
    files = ("gradle.properties", "build.gradle",
             "settings.gradle", "src/main/resources/fabric.mod.json")
    records = []
    for relative in files:
        path = root / relative
        records.append([relative, _sha(path.read_bytes()) if path.is_file() else None])
    return _sha(json.dumps(records, separators=(",", ":")).encode())


def _path(root: Path, target_path: str) -> Path:
    if not target_path.startswith(("src/main/java/", "src/client/java/")):
        raise ValueError("COMPILER_PATCH_TARGET_INVALID")
    target = (root / target_path).resolve()
    if not target.is_relative_to(root) or not target_path.endswith(".java"):
        raise ValueError("COMPILER_PATCH_TARGET_INVALID")
    key = _sha(target_path.encode("utf-8"))
    return root / ".mmm" / "compiled_feature_patches" / (key + ".json")


def persist_compiled_repair(
    project_root: str | Path, target_path: str,
    *,
    original_sha256: str, compiled_source: str,
) -> dict:
    root = Path(project_root).resolve()
    path = _path(root, target_path)
    source = compiled_source.encode("utf-8")
    _parse(compiled_source)
    if len(source) > 1_000_000 or len(original_sha256) != 64:
        raise ValueError("COMPILER_PATCH_SOURCE_INVALID")
    current = (root / target_path).resolve()
    if not current.is_file() or _sha(current.read_bytes()) != _sha(source):
        raise ValueError("COMPILER_PATCH_NOT_CURRENT_SOURCE")
    envelope = {
        "schema_version": "mmm/compiled-feature-patch-v1",
        "target_path": target_path,
        "original_sha256": original_sha256,
        "compiled_sha256": _sha(source),
        "project_contract_sha256": _project_hash(root),
        "compiled_source": compiled_source,
        "runtime_verified": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return {key: envelope[key] for key in (
        "schema_version", "target_path", "original_sha256",
        "compiled_sha256", "project_contract_sha256", "runtime_verified"
    )}


def replay_compiled_repair(project_root: str | Path, target_path: str, source: str) -> str:
    root = Path(project_root).resolve()
    path = _path(root, target_path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_200_000:
        return source
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return source
    if (row.get("schema_version") != "mmm/compiled-feature-patch-v1"
            or row.get("target_path") != target_path
            or row.get("project_contract_sha256") != _project_hash(root)):
        return source
    candidate = row.get("compiled_source")
    if not isinstance(candidate, str) or _sha(candidate.encode()) != row.get("compiled_sha256"):
        return source
    current_sha = _sha(source.encode("utf-8"))
    if current_sha == row["compiled_sha256"]:
        return source
    if current_sha != row.get("original_sha256"):
        return source
    _parse(candidate)
    return candidate
