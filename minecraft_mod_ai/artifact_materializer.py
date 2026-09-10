from __future__ import annotations

"""Deterministic file materializer with before/after SHA-256 receipts.

Supports four materialization modes:
1. whole_file: Writes complete source or resource file.
2. java_patch: Inserts code at exact anchor coordinates (e.g. /* MMM:properties:raw_lunite */).
3. json_merge: Merges JSON fragments (e.g. lang file entries).
4. binary_asset: Writes raw binary data (e.g. textures).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from .artifact_job import ArtifactJob


class MaterializeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializeReceipt:
    target_path: str
    operation: str
    before_sha256: str | None
    after_sha256: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_path(path: Path | str, base_dir: Path | None = None) -> Path:
    p = Path(path)
    if base_dir is not None and not p.is_absolute():
        p = base_dir / p
    return p


def materialize_whole_file(
    target_path: Path | str,
    content: str | bytes,
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Write complete file to target path and return SHA256 receipt."""
    p = _resolve_path(target_path, base_dir)
    data = content.encode("utf-8") if isinstance(content, str) else content

    before_sha: str | None = None
    if p.is_file():
        before_sha = _sha256(p.read_bytes())

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    after_sha = _sha256(data)

    return MaterializeReceipt(
        target_path=str(p),
        operation="whole_file",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"bytes_written": len(data)},
    )


def materialize_java_patch(
    target_path: Path | str,
    anchor: str,
    patch: str,
    *,
    base_dir: Path | None = None,
    keep_anchor: bool = True,
) -> MaterializeReceipt:
    """Patch Java source at exact anchor coordinate."""
    p = _resolve_path(target_path, base_dir)
    if not p.is_file():
        raise MaterializeError(f"PATCH_TARGET_NOT_FOUND: Target file {p} does not exist for anchor {anchor!r}")

    original_bytes = p.read_bytes()
    before_sha = _sha256(original_bytes)
    original_text = original_bytes.decode("utf-8")

    if anchor not in original_text:
        raise MaterializeError(f"ANCHOR_NOT_FOUND: Anchor {anchor!r} not found in {p}")

    if keep_anchor:
        # Check indentation of the anchor line
        replacement = f"{patch.strip()}\n    {anchor}"
    else:
        replacement = patch.strip()

    updated_text = original_text.replace(anchor, replacement, 1)
    updated_bytes = updated_text.encode("utf-8")
    p.write_bytes(updated_bytes)
    after_sha = _sha256(updated_bytes)

    return MaterializeReceipt(
        target_path=str(p),
        operation="java_patch",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"anchor": anchor, "keep_anchor": keep_anchor},
    )


def _deep_merge_dicts(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in updates.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, Mapping):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def materialize_json_merge(
    target_path: Path | str,
    fragment: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Merge JSON fragment into existing JSON resource or create a new one."""
    p = _resolve_path(target_path, base_dir)
    before_sha: str | None = None
    existing_data: dict[str, Any] = {}

    if p.is_file():
        raw_bytes = p.read_bytes()
        before_sha = _sha256(raw_bytes)
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(parsed, dict):
                existing_data = parsed
        except Exception:
            existing_data = {}

    merged = _deep_merge_dicts(existing_data, fragment)
    formatted = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
    updated_bytes = formatted.encode("utf-8")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(updated_bytes)
    after_sha = _sha256(updated_bytes)

    return MaterializeReceipt(
        target_path=str(p),
        operation="json_merge",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"keys_count": len(merged)},
    )


def materialize_binary_asset(
    target_path: Path | str,
    data: bytes,
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Write binary asset (such as PNG texture)."""
    p = _resolve_path(target_path, base_dir)
    before_sha: str | None = None
    if p.is_file():
        before_sha = _sha256(p.read_bytes())

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    after_sha = _sha256(data)

    return MaterializeReceipt(
        target_path=str(p),
        operation="binary_asset",
        before_sha256=before_sha,
        after_sha256=after_sha,
        status="SUCCESS",
        details={"bytes_written": len(data)},
    )


def materialize_job_output(
    job: ArtifactJob,
    rendered_content: Any,
    *,
    base_dir: Path | None = None,
) -> MaterializeReceipt:
    """Materialize rendered ArtifactJob output according to its target and anchor."""
    target_file = job.target_path
    if not target_file:
        raise MaterializeError(f"JOB_NO_TARGET: Job {job.job_id} has no target_path")

    # If anchor is specified, perform java_patch
    if job.anchor:
        # Settings properties are per-item and should replace the anchor
        is_property_anchor = "properties:" in job.anchor
        patch_str = str(rendered_content)
        return materialize_java_patch(
            target_file,
            job.anchor,
            patch_str,
            base_dir=base_dir,
            keep_anchor=not is_property_anchor,
        )

    # If it's a JSON fragment (e.g. lang file), perform json_merge
    if target_file.endswith(".json") and (isinstance(rendered_content, Mapping) or "lang" in job.template_id):
        if isinstance(rendered_content, str):
            fragment = json.loads(rendered_content)
        else:
            fragment = dict(rendered_content)
        return materialize_json_merge(target_file, fragment, base_dir=base_dir)

    # Otherwise whole_file
    if isinstance(rendered_content, bytes):
        return materialize_binary_asset(target_file, rendered_content, base_dir=base_dir)

    content_str = (
        json.dumps(rendered_content, indent=2, ensure_ascii=False) + "\n"
        if isinstance(rendered_content, (dict, list))
        else str(rendered_content)
    )
    return materialize_whole_file(target_file, content_str, base_dir=base_dir)
