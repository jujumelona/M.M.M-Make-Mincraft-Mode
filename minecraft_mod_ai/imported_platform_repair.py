from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "mmm/imported-platform-repair-v1"
RELATIVE_PATH = Path(".minecraft_ai/imported-platform-repair.json")
_SHA256 = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


def _canonical_sha256(value: str) -> str:
    match = _SHA256.fullmatch(str(value).strip())
    if match is None:
        raise ValueError("Imported platform repair requires the bound source SHA-256.")
    return "sha256:" + match.group(1)


def marker_path(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    return root / RELATIVE_PATH


def write_marker(
    project_root: str | Path,
    *,
    adapter: Any,
    archive_sha256: str,
    reason: str,
) -> Path:
    digest = _canonical_sha256(archive_sha256)
    root = Path(project_root).expanduser().resolve()
    target = marker_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError("Imported platform repair metadata directory is unsafe.")
    payload = {
        "schema_version": SCHEMA,
        "status": "REPAIR_REQUIRED",
        "expected": {
            "adapter_id": str(adapter.adapter_id),
            "minecraft_version": str(adapter.minecraft_version),
            "loader": str(adapter.loader),
            "yarn_mappings": str(adapter.yarn_mappings),
            "java_version": str(adapter.java_version),
            "fabric_loader": str(adapter.fabric_loader),
            "fabric_api": str(adapter.fabric_api),
            "fabric_loom": str(adapter.fabric_loom),
            "gradle": str(adapter.gradle),
        },
        "archive_sha256": digest,
        "reason": str(reason)[:2000],
        "authority": "repair-entry-only",
        "release_evidence": False,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_valid_marker(
    project_root: str | Path,
    *,
    adapter: Any,
    archive_sha256: str | None = None,
) -> dict[str, Any] | None:
    root = Path(project_root).expanduser().resolve()
    target = marker_path(root)
    if not target.is_file() or target.is_symlink():
        return None
    try:
        target.resolve().relative_to(root)
    except ValueError:
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = payload.get("expected")
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "REPAIR_REQUIRED"
        or payload.get("authority") != "repair-entry-only"
        or payload.get("release_evidence") is not False
        or not isinstance(expected, dict)
    ):
        return None
    required = {
        "adapter_id": str(adapter.adapter_id),
        "minecraft_version": str(adapter.minecraft_version),
        "loader": str(adapter.loader),
        "yarn_mappings": str(adapter.yarn_mappings),
        "java_version": str(adapter.java_version),
        "fabric_loader": str(adapter.fabric_loader),
        "fabric_api": str(adapter.fabric_api),
        "fabric_loom": str(adapter.fabric_loom),
        "gradle": str(adapter.gradle),
    }
    if any(str(expected.get(key, "")) != value for key, value in required.items()):
        return None
    digest = payload.get("archive_sha256")
    if not isinstance(digest, str):
        return None
    try:
        digest = _canonical_sha256(digest)
    except ValueError:
        return None
    if archive_sha256 is not None:
        try:
            expected_digest = _canonical_sha256(archive_sha256)
        except ValueError:
            return None
        if digest != expected_digest:
            return None
    return {**payload, "archive_sha256": digest}


def clear_marker(project_root: str | Path) -> None:
    target = marker_path(project_root)
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError("Imported platform repair marker is unsafe.")
        target.unlink()
