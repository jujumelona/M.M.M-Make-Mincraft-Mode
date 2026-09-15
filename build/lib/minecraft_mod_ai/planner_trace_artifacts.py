"""Unabridged diagnostic artifacts referenced by bounded console/journal events."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def repository_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
        return {"git_commit": sha, "tracked_worktree_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unavailable"}


def _redacted(value: Any, seen: set[int]) -> Any:
    from .root_cause_trace import _secret_key

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if id(value) in seen:
        return "<cycle>"
    seen.add(id(value))
    try:
        if isinstance(value, Mapping):
            return {
                str(key): "<redacted>" if _secret_key(key) else _redacted(child, seen)
                for key, child in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return [_redacted(child, seen) for child in value]
        return str(value)
    finally:
        seen.remove(id(value))


def save_trace_artifact(
    value: Any,
    directory: Path,
    *,
    sync: bool = True,
) -> dict[str, Any]:
    """Persist one deduplicated trace artifact.

    ``sync`` controls the durability barrier, not whether the artifact is written. Failure
    paths keep ``sync=True``; routine success events may set it to ``False`` so tracing does
    not force a disk barrier on every model/tool boundary.
    """

    data = json.dumps(
        _redacted(value, set()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".json")
    if not path.exists():
        temporary = directory / ("." + digest + "-" + uuid.uuid4().hex)
        try:
            with temporary.open("wb") as stream:
                stream.write(data)
                stream.flush()
                if sync:
                    os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": digest, "bytes": len(data)}
