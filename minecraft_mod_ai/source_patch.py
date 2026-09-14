from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mod_output_scope import ModOutputScopeError, validate_mod_output_path
from .project_mutation import FileChange, mutation_owner
from .residual_generation_contract import (
    ResidualContractLoadError,
    ResidualGenerationContract,
    load_residual_generation_contracts,
    validate_residual_write_against_contracts,
)

_WORKSPACE_IMPACTS = frozenset({"unchanged", "rolled_back", "drift", "uncertain"})
_COMMIT_POOL_LOCK = threading.RLock()
_COMMIT_POOL: ThreadPoolExecutor | None = None


def _global_commit_worker_count() -> int:
    raw = os.environ.get("MMM_SOURCE_PATCH_GLOBAL_WORKERS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise SourcePatchError("MMM_SOURCE_PATCH_GLOBAL_WORKERS must be a positive integer") from exc
        if value < 1:
            raise SourcePatchError("MMM_SOURCE_PATCH_GLOBAL_WORKERS must be a positive integer")
        return min(32, value)
    return min(16, max(2, (os.cpu_count() or 1) * 2))


def _shared_commit_pool() -> ThreadPoolExecutor:
    global _COMMIT_POOL
    with _COMMIT_POOL_LOCK:
        if _COMMIT_POOL is None:
            _COMMIT_POOL = ThreadPoolExecutor(
                max_workers=_global_commit_worker_count(),
                thread_name_prefix="mmm_source_patch_commit",
            )
        return _COMMIT_POOL


class SourcePatchError(RuntimeError):
    """Raised when a bounded source patch cannot be validated or committed.

    ``workspace_impact`` is a host-owned transaction fact consumed by the causal
    state ledger. It distinguishes harmless pre-write failures from stale-snapshot
    drift and from failures where rollback could not prove the workspace state.
    """

    def __init__(self, message: str, *, workspace_impact: str = "unchanged") -> None:
        impact = str(workspace_impact).strip().casefold()
        if impact not in _WORKSPACE_IMPACTS:
            raise ValueError(f"Unsupported source-patch workspace impact: {workspace_impact!r}")
        self.workspace_impact = impact
        super().__init__(message)

    def __str__(self) -> str:
        return f"{super().__str__()} [workspace_impact={self.workspace_impact}]"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _commit_worker_count(count: int) -> int:
    if count <= 1:
        return 1
    raw = os.environ.get("MMM_SOURCE_PATCH_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
    else:
        requested = min(16, max(2, (os.cpu_count() or 1) * 2))
    return max(1, min(int(count), 32, requested))


def _commit_staged_path(path: Path, after: bytes | None) -> None:
    """Commit one already-validated path without acquiring another project lock."""

    if after is None:
        path.unlink()
        return
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


@dataclass(frozen=True)
class PatchReceipt:
    path: str
    operation: str
    before_sha256: str | None
    after_sha256: str | None

    @property
    def changed(self) -> bool:
        return self.before_sha256 != self.after_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "operation": self.operation,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


class TransactionalSourcePatcher:
    """Apply exact, hash-guarded text patches inside one project root."""

    _OPERATION_FIELDS = {
        "create": frozenset({"operation", "path", "content"}),
        "replace": frozenset({"operation", "path", "expected_sha256", "content"}),
        "edit": frozenset({"operation", "path", "expected_sha256", "replacements"}),
        "delete": frozenset({"operation", "path", "expected_sha256"}),
    }
    _CONDITIONAL_FIELDS = frozenset({"content", "expected_sha256", "replacements"})

    def __init__(
        self,
        project_root: str | Path,
        *,
        residual_contracts: Iterable[ResidualGenerationContract] | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir() or self.project_root.is_symlink():
            raise SourcePatchError(f"Project root is not a real directory: {self.project_root}")
        try:
            self._residual_contracts = tuple(
                residual_contracts
                if residual_contracts is not None
                else load_residual_generation_contracts(self.project_root)
            )
        except ResidualContractLoadError as exc:
            raise SourcePatchError(f"Residual write policy is invalid: {exc}") from exc

    def _normalized_transaction(
        self,
        operations: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = [self._normalize(item) for item in operations]
        if not normalized:
            raise SourcePatchError("At least one patch operation is required.")
        paths = [item["path"] for item in normalized]
        if len(paths) != len(set(paths)):
            raise SourcePatchError("A patch transaction may touch each path only once.")
        return normalized

    def apply(self, operations: Iterable[dict[str, Any]]) -> dict[str, Any]:
        normalized = self._normalized_transaction(operations)
        paths = [item["path"] for item in normalized]
        owner = mutation_owner(self.project_root)
        with owner.transaction(paths):
            try:
                return self._apply_locked(normalized)
            except SourcePatchError as exc:
                self._invalidate_on_uncertain_state(owner, exc)
                raise

    @staticmethod
    def _invalidate_on_uncertain_state(owner: Any, exc: SourcePatchError) -> None:
        if exc.workspace_impact in {"drift", "uncertain"}:
            owner.invalidate()

    def _validate_residual_contract(self, item: dict[str, Any], before: bytes | None) -> None:
        if not self._residual_contracts:
            return
        try:
            validate_residual_write_against_contracts(
                item["path"],
                sha256_bytes(before) if before is not None else None,
                self._residual_contracts,
            )
        except PermissionError as exc:
            raise SourcePatchError(f"RESIDUAL_WRITE_CONTRACT: {exc}") from exc

    def _operation_result(
        self,
        item: dict[str, Any],
        *,
        exists: bool,
        before: bytes | None,
    ) -> bytes | None:
        operation = item["operation"]
        if operation == "create":
            if exists:
                raise SourcePatchError(f"Create target already exists: {item['path']}")
            return item["content"].encode("utf-8")
        if not exists:
            raise SourcePatchError(f"Patch target does not exist: {item['path']}")
        actual = sha256_bytes(before or b"")
        if item.get("expected_sha256") != actual:
            raise SourcePatchError(
                f"SHA-256 precondition failed for {item['path']}: "
                f"{actual} != {item.get('expected_sha256')}",
                workspace_impact="drift",
            )
        if operation == "replace":
            return item["content"].encode("utf-8")
        if operation == "edit":
            return self._edit(before or b"", item).encode("utf-8")
        if operation == "delete":
            return None
        raise SourcePatchError(f"Unsupported operation: {operation}")

    def _validate_json_result(self, item: dict[str, Any], after: bytes | None) -> None:
        if after is None or not item["path"].endswith(".json"):
            return
        from .typed_resources import JsonResource, ResourceValidationError, resource_kind

        try:
            JsonResource.parse(after.decode("utf-8"), kind=resource_kind(item["path"]))
        except (UnicodeError, ResourceValidationError) as exc:
            raise SourcePatchError(f"Invalid JSON output {item['path']}: {exc}") from exc

    @staticmethod
    def _preserve_equivalent_bytes(before: bytes | None, after: bytes | None) -> bytes | None:
        if before == after:
            return before
        if before is None or after is None:
            return after
        if before.replace(b"\r\n", b"\n") == after.replace(b"\r\n", b"\n"):
            return before
        return after

    def _stage_one(
        self,
        item: dict[str, Any],
        validated_parents: set[Path],
    ) -> tuple[Path, bytes | None, bytes | None, PatchReceipt]:
        path = self._path(
            item["path"],
            allow_missing=item["operation"] == "create",
            validated_parents=validated_parents,
        )
        exists = path.exists()
        if exists and (not path.is_file() or path.is_symlink()):
            raise SourcePatchError(f"Patch target is not a regular file: {item['path']}")
        before = path.read_bytes() if exists else None
        self._validate_residual_contract(item, before)
        after = self._operation_result(item, exists=exists, before=before)
        self._validate_json_result(item, after)
        after = self._preserve_equivalent_bytes(before, after)
        receipt = PatchReceipt(
            path=item["path"],
            operation=item["operation"],
            before_sha256=sha256_bytes(before) if before is not None else None,
            after_sha256=sha256_bytes(after) if after is not None else None,
        )
        return path, before, after, receipt

    def _stage_transaction(
        self,
        normalized: Iterable[dict[str, Any]],
    ) -> tuple[dict[Path, bytes | None], dict[Path, bytes | None], list[PatchReceipt]]:
        staged: dict[Path, bytes | None] = {}
        originals: dict[Path, bytes | None] = {}
        receipts: list[PatchReceipt] = []
        validated_parents: set[Path] = {self.project_root}
        for item in normalized:
            path, before, after, receipt = self._stage_one(item, validated_parents)
            originals[path] = before
            staged[path] = after
            receipts.append(receipt)
        return staged, originals, receipts

    @staticmethod
    def _changed_staged_paths(
        staged: dict[Path, bytes | None],
        originals: dict[Path, bytes | None],
    ) -> list[tuple[Path, bytes | None]]:
        return [(path, after) for path, after in staged.items() if originals[path] != after]

    @staticmethod
    def _serial_commit(
        ordered_staged: list[tuple[Path, bytes | None]],
    ) -> tuple[set[Path], dict[Path, BaseException]]:
        committed: set[Path] = set()
        errors: dict[Path, BaseException] = {}
        for path, after in ordered_staged:
            try:
                _commit_staged_path(path, after)
            except BaseException as exc:
                errors[path] = exc
                break
            committed.add(path)
        return committed, errors

    @staticmethod
    def _parallel_commit(
        ordered_staged: list[tuple[Path, bytes | None]],
        workers: int,
    ) -> tuple[set[Path], dict[Path, BaseException]]:
        committed: set[Path] = set()
        errors: dict[Path, BaseException] = {}
        permits = threading.BoundedSemaphore(workers)

        def commit_with_permit(path: Path, after: bytes | None) -> None:
            with permits:
                _commit_staged_path(path, after)

        pool = _shared_commit_pool()
        futures = {
            pool.submit(commit_with_permit, path, after): path
            for path, after in ordered_staged
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except BaseException as exc:
                errors[path] = exc
            else:
                committed.add(path)
        return committed, errors

    def _commit_transaction(
        self,
        ordered_staged: list[tuple[Path, bytes | None]],
    ) -> tuple[set[Path], dict[Path, BaseException]]:
        workers = _commit_worker_count(len(ordered_staged))
        if workers <= 1:
            return self._serial_commit(ordered_staged)
        return self._parallel_commit(ordered_staged, workers)

    def _rollback_committed(
        self,
        committed_order: list[Path],
        originals: dict[Path, bytes | None],
    ) -> list[tuple[Path, BaseException]]:
        rollback_errors: list[tuple[Path, BaseException]] = []
        for path in reversed(committed_order):
            original = originals[path]
            try:
                if original is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(original)
            except BaseException as exc:
                rollback_errors.append((path, exc))
        return rollback_errors

    def _raise_commit_failure(
        self,
        ordered_staged: list[tuple[Path, bytes | None]],
        originals: dict[Path, bytes | None],
        committed: set[Path],
        errors: dict[Path, BaseException],
    ) -> None:
        if not errors:
            return
        committed_order = [path for path, _after in ordered_staged if path in committed]
        rollback_errors = self._rollback_committed(committed_order, originals)
        first_error = next(errors[path] for path, _after in ordered_staged if path in errors)
        if rollback_errors:
            rollback_path, rollback_error = rollback_errors[0]
            relative = rollback_path.relative_to(self.project_root).as_posix()
            raise SourcePatchError(
                "Patch transaction failed and rollback was incomplete for "
                f"{relative}: {rollback_error}",
                workspace_impact="uncertain",
            ) from first_error
        raise SourcePatchError(
            f"Patch transaction rolled back: {first_error}",
            workspace_impact="rolled_back",
        ) from first_error

    def _receipt(self, receipts: list[PatchReceipt]) -> dict[str, Any]:
        changed_paths = [receipt.path for receipt in receipts if receipt.changed]
        changes = mutation_owner(self.project_root).committed(
            FileChange(
                receipt.path,
                receipt.operation,
                receipt.before_sha256,
                receipt.after_sha256,
            )
            for receipt in receipts
        )
        return {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "APPLIED" if changed_paths else "UNCHANGED",
            "project_root": str(self.project_root),
            "changed_paths": changed_paths,
            "operations": [receipt.to_dict() for receipt in receipts],
            "change_set": changes.to_dict(),
        }

    def _apply_locked(self, operations: Iterable[dict[str, Any]]) -> dict[str, Any]:
        normalized = self._normalized_transaction(operations)
        staged, originals, receipts = self._stage_transaction(normalized)
        ordered_staged = self._changed_staged_paths(staged, originals)
        committed, errors = self._commit_transaction(ordered_staged)
        self._raise_commit_failure(ordered_staged, originals, committed, errors)
        return self._receipt(receipts)

    def snapshot(self, relative_paths: Iterable[str]) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        validated_parents: set[Path] = {self.project_root}
        for relative in relative_paths:
            path = self._path(
                relative,
                allow_missing=False,
                validated_parents=validated_parents,
            )
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

    @classmethod
    def _canonicalize_known_sibling_fields(
        cls,
        operation: str,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Drop only fields that are valid for a different source operation."""

        allowed = cls._OPERATION_FIELDS.get(operation)
        canonical = dict(value)
        if allowed is None:
            return canonical
        for field in cls._CONDITIONAL_FIELDS - allowed:
            canonical.pop(field, None)
        return canonical

    def _normalize(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise SourcePatchError("Every patch operation must be an object.")
        operation = value.get("operation")
        if operation not in self._OPERATION_FIELDS:
            raise SourcePatchError(f"Unsupported patch operation: {operation!r}")
        value = self._canonicalize_known_sibling_fields(operation, value)
        relative = value.get("path")
        if not isinstance(relative, str) or not relative.strip():
            raise SourcePatchError("Patch path must be a non-empty string.")
        relative = relative.strip()
        try:
            validate_mod_output_path(relative)
        except ModOutputScopeError as exc:
            raise SourcePatchError(str(exc)) from exc
        normalized: dict[str, Any] = {"operation": operation, "path": relative}
        if operation == "create":
            if set(value) - self._OPERATION_FIELDS["create"]:
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
            if set(value) - self._OPERATION_FIELDS["replace"]:
                raise SourcePatchError(f"Unknown replace fields for {relative}")
            if not isinstance(value.get("content"), str):
                raise SourcePatchError(f"Replace content must be text: {relative}")
            normalized["content"] = value["content"]
        elif operation == "edit":
            if set(value) - self._OPERATION_FIELDS["edit"]:
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
            if set(value) - self._OPERATION_FIELDS["delete"]:
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

    def _path(
        self,
        relative: str,
        *,
        allow_missing: bool,
        validated_parents: set[Path] | None = None,
    ) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SourcePatchError(f"Unsafe patch path: {relative}")
        target = self.project_root / candidate
        if target == self.project_root:
            raise SourcePatchError("The project root itself cannot be patched.")
        current = self.project_root
        for part in candidate.parts[:-1]:
            current = current / part
            if validated_parents is not None and current in validated_parents:
                continue
            if current.is_symlink():
                raise SourcePatchError(f"Patch parent contains a symlink: {relative}")
            if current.exists() and not current.is_dir():
                raise SourcePatchError(f"Patch parent is not a directory: {relative}")
            if validated_parents is not None:
                validated_parents.add(current)
        if not allow_missing and not target.exists():
            raise SourcePatchError(f"Patch target does not exist: {relative}")
        return target
