from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .project_write_lock import project_write_lock


class SourcePatchError(RuntimeError):
    """Raised when a bounded source patch cannot be validated or committed."""


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class PatchReceipt:
    path: str
    operation: str
    before_sha256: str | None
    after_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "operation": self.operation,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


class TransactionalSourcePatcher:
    """Apply exact, hash-guarded text patches inside one project root.

    Operations are fully validated in memory before any file is changed. Writes use
    atomic ``os.replace`` and all touched files are rolled back if any commit step
    fails. Symlinks, path traversal and broad directory deletion are rejected.
    Concurrent transactions for the same project root are serialized so rollback from
    one transaction cannot overwrite another transaction's commit.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir() or self.project_root.is_symlink():
            raise SourcePatchError(f"Project root is not a real directory: {self.project_root}")

    def apply(self, operations: Iterable[dict[str, Any]]) -> dict[str, Any]:
        # Keep the lock around validation/staging as well as commit. expected_sha256
        # must be checked against the same project state that is ultimately mutated.
        # The lock is re-entrant because higher-level generators may already hold it
        # while performing an atomic read/merge/write sequence.
        with project_write_lock(self.project_root):
            return self._apply_locked(operations)

    def _apply_locked(self, operations: Iterable[dict[str, Any]]) -> dict[str, Any]:
        normalized = [self._normalize(item) for item in operations]
        if not normalized:
            raise SourcePatchError("At least one patch operation is required.")
        paths = [item["path"] for item in normalized]
        if len(paths) != len(set(paths)):
            raise SourcePatchError("A patch transaction may touch each path only once.")

        staged: dict[Path, bytes | None] = {}
        originals: dict[Path, bytes | None] = {}
        receipts: list[PatchReceipt] = []
        for item in normalized:
            path = self._path(item["path"], allow_missing=item["operation"] == "create")
            exists = path.exists()
            if exists and (not path.is_file() or path.is_symlink()):
                raise SourcePatchError(f"Patch target is not a regular file: {item['path']}")
            before = path.read_bytes() if exists else None
            originals[path] = before
            expected = item.get("expected_sha256")
            if item["operation"] == "create":
                if exists:
                    raise SourcePatchError(f"Create target already exists: {item['path']}")
                after = item["content"].encode("utf-8")
            else:
                if not exists:
                    raise SourcePatchError(f"Patch target does not exist: {item['path']}")
                actual = sha256_bytes(before or b"")
                if expected != actual:
                    raise SourcePatchError(
                        f"SHA-256 precondition failed for {item['path']}: {actual} != {expected}"
                    )
                if item["operation"] == "replace":
                    after = item["content"].encode("utf-8")
                elif item["operation"] == "edit":
                    after = self._edit(before or b"", item).encode("utf-8")
                elif item["operation"] == "delete":
                    after = None
                else:  # pragma: no cover - normalized above
                    raise SourcePatchError(f"Unsupported operation: {item['operation']}")
            if before == after:
                raise SourcePatchError(f"Patch operation makes no change: {item['path']}")
            staged[path] = after
            receipts.append(
                PatchReceipt(
                    path=item["path"],
                    operation=item["operation"],
                    before_sha256=sha256_bytes(before) if before is not None else None,
                    after_sha256=sha256_bytes(after) if after is not None else None,
                )
            )

        committed: list[Path] = []
        try:
            for path, after in staged.items():
                if after is None:
                    path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
                    try:
                        with os.fdopen(fd, "wb") as stream:
                            stream.write(after)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(tmp_name, path)
                    finally:
                        if os.path.exists(tmp_name):
                            os.unlink(tmp_name)
                committed.append(path)
        except Exception as exc:
            for path in reversed(committed):
                original = originals[path]
                if original is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(original)
            raise SourcePatchError(f"Patch transaction rolled back: {exc}") from exc

        return {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED",
            "project_root": str(self.project_root),
            "operations": [receipt.to_dict() for receipt in receipts],
        }

    def snapshot(self, relative_paths: Iterable[str]) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for relative in relative_paths:
            path = self._path(relative, allow_missing=False)
            if not path.is_file() or path.is_symlink():
                raise SourcePatchError(f"Snapshot target is not a regular file: {relative}")
            text = path.read_text(encoding="utf-8")
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "content": text,
                }
            )
        return {
            "schema_version": "mmm/source-snapshot-v1",
            "project_root": str(self.project_root),
            "files": files,
        }

    def _normalize(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise SourcePatchError("Every patch operation must be an object.")
        operation = value.get("operation")
        if operation not in {"create", "replace", "edit", "delete"}:
            raise SourcePatchError(f"Unsupported patch operation: {operation!r}")
        relative = value.get("path")
        if not isinstance(relative, str) or not relative.strip():
            raise SourcePatchError("Patch path must be a non-empty string.")
        normalized: dict[str, Any] = {"operation": operation, "path": relative.strip()}
        if operation == "create":
            if set(value) - {"operation", "path", "content"}:
                raise SourcePatchError(f"Unknown create fields for {relative}")
            if not isinstance(value.get("content"), str):
                raise SourcePatchError(f"Create content must be text: {relative}")
            normalized["content"] = value["content"]
            return normalized
        expected = value.get("expected_sha256")
        if not isinstance(expected, str) or not expected.startswith("sha256:") or len(expected) != 71:
            raise SourcePatchError(f"A valid expected_sha256 is required for {relative}")
        normalized["expected_sha256"] = expected
        if operation == "replace":
            if set(value) - {"operation", "path", "expected_sha256", "content"}:
                raise SourcePatchError(f"Unknown replace fields for {relative}")
            if not isinstance(value.get("content"), str):
                raise SourcePatchError(f"Replace content must be text: {relative}")
            normalized["content"] = value["content"]
        elif operation == "edit":
            if set(value) - {"operation", "path", "expected_sha256", "replacements"}:
                raise SourcePatchError(f"Unknown edit fields for {relative}")
            replacements = value.get("replacements")
            if not isinstance(replacements, list) or not replacements:
                raise SourcePatchError(f"Edit replacements must be a non-empty list: {relative}")
            clean: list[dict[str, Any]] = []
            for replacement in replacements:
                if not isinstance(replacement, dict) or set(replacement) - {"old", "new", "count"}:
                    raise SourcePatchError(f"Invalid replacement object for {relative}")
                old, new = replacement.get("old"), replacement.get("new")
                count = replacement.get("count", 1)
                if not isinstance(old, str) or not old:
                    raise SourcePatchError(f"Replacement old text must be non-empty: {relative}")
                if not isinstance(new, str) or type(count) is not int or count < 1:
                    raise SourcePatchError(f"Invalid replacement new/count for {relative}")
                clean.append({"old": old, "new": new, "count": count})
            normalized["replacements"] = clean
        else:
            if set(value) - {"operation", "path", "expected_sha256"}:
                raise SourcePatchError(f"Unknown delete fields for {relative}")
        return normalized

    def _edit(self, raw: bytes, item: dict[str, Any]) -> str:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourcePatchError(f"Edit target is not UTF-8 text: {item['path']}") from exc
        for replacement in item["replacements"]:
            found = text.count(replacement["old"])
            if found != replacement["count"]:
                raise SourcePatchError(
                    f"Replacement precondition failed for {item['path']}: expected "
                    f"{replacement['count']} exact matches, found {found}."
                )
            text = text.replace(replacement["old"], replacement["new"], replacement["count"])
        return text

    def _path(self, relative: str, *, allow_missing: bool) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SourcePatchError(f"Unsafe patch path: {relative}")
        target = (self.project_root / candidate).resolve(strict=False)
        try:
            target.relative_to(self.project_root)
        except ValueError as exc:
            raise SourcePatchError(f"Patch path escaped the project root: {relative}") from exc
        if target == self.project_root:
            raise SourcePatchError("The project root itself cannot be patched.")
        current = self.project_root
        for part in candidate.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise SourcePatchError(f"Patch parent contains a symlink: {relative}")
            if current.exists() and not current.is_dir():
                raise SourcePatchError(f"Patch parent is not a directory: {relative}")
        if not allow_missing and not target.exists():
            raise SourcePatchError(f"Patch target does not exist: {relative}")
        return target
