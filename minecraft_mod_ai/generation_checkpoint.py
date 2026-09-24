from __future__ import annotations

"""Persistent custom-module staging/checkpoint ownership.

The custom-module generator decides *what* to generate. This module owns the
filesystem transaction around resumable staged work: identity, leases, snapshots,
checkpoint persistence/rebase, staged-operation collection, and durable cleanup.
"""

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

from .custom_module_errors import CustomModuleGenerationError
from .filesystem_copy import reflink_or_copy
from .host_grounding import custom_module_path_allowed, custom_module_path_protected
from .research_validation_fingerprint_performance import content_digest
from .source_patch import SourcePatchError, TransactionalSourcePatcher

_STAGE_IGNORED_DIRS = {".git", ".gradle", ".minecraft_ai", "build", "run"}
_CHECKPOINT_DIRECTORY = ".mmm-custom-checkpoints"
_CHECKPOINT_SCHEMA = "mmm/custom-module-checkpoint-v2"
_CHECKPOINT_KEY = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_STRATEGY_EPOCH = "mmm/custom-candidate-strategy-v1"
_CHECKPOINT_ACTIVE_LOCK = threading.RLock()
_CHECKPOINT_ACTIVE_PATHS: set[Path] = set()
_CHECKPOINT_LEASE_SCOPE = threading.local()
_CHECKPOINT_PERSISTENCE_SCOPE = threading.local()


class _GenerationCheckpointLease:
    """Exclusive in-process ownership of one persistent staged workspace."""

    def __init__(self, checkpoint_root: Path) -> None:
        self.checkpoint_root = checkpoint_root.resolve()
        self._closed = False
        with _CHECKPOINT_ACTIVE_LOCK:
            if self.checkpoint_root in _CHECKPOINT_ACTIVE_PATHS:
                raise CustomModuleGenerationError(
                    "An identical custom-module checkpoint is already active."
                )
            _CHECKPOINT_ACTIVE_PATHS.add(self.checkpoint_root)

    def close(self) -> None:
        with _CHECKPOINT_ACTIVE_LOCK:
            if self._closed:
                return
            _CHECKPOINT_ACTIVE_PATHS.discard(self.checkpoint_root)
            self._closed = True


def _checkpoint_lease_scoped(method):
    @wraps(method)
    def guarded(*args: Any, **kwargs: Any):
        previous = getattr(_CHECKPOINT_LEASE_SCOPE, "leases", None)
        leases: list[_GenerationCheckpointLease] = []
        _CHECKPOINT_LEASE_SCOPE.leases = leases
        try:
            return method(*args, **kwargs)
        finally:
            for lease in reversed(leases):
                lease.close()
            if previous is None:
                try:
                    del _CHECKPOINT_LEASE_SCOPE.leases
                except AttributeError:
                    pass
            else:
                _CHECKPOINT_LEASE_SCOPE.leases = previous

    return guarded


def _track_checkpoint_lease(lease: _GenerationCheckpointLease) -> None:
    leases = getattr(_CHECKPOINT_LEASE_SCOPE, "leases", None)
    if not isinstance(leases, list):
        lease.close()
        raise CustomModuleGenerationError(
            "Checkpoint lease was opened outside the guarded generation scope."
        )
    leases.append(lease)


def _transfer_checkpoint_lease(lease: _GenerationCheckpointLease) -> None:
    leases = getattr(_CHECKPOINT_LEASE_SCOPE, "leases", None)
    if not isinstance(leases, list) or lease not in leases:
        raise CustomModuleGenerationError(
            "Checkpoint lease is not owned by the active generation scope."
        )
    leases.remove(lease)


@contextmanager
def _active_checkpoint_persistence(
    checkpoint_root: Path,
    staged_root: Path,
    identity_sha256: str,
):
    previous = getattr(_CHECKPOINT_PERSISTENCE_SCOPE, "state", None)
    _CHECKPOINT_PERSISTENCE_SCOPE.state = (
        checkpoint_root,
        staged_root.resolve(),
        identity_sha256,
    )
    try:
        yield
    finally:
        if previous is None:
            try:
                del _CHECKPOINT_PERSISTENCE_SCOPE.state
            except AttributeError:
                pass
        else:
            _CHECKPOINT_PERSISTENCE_SCOPE.state = previous


def persist_active_generation_checkpoint(project_root: str | Path) -> bool:
    """Persist the exact active staged workspace after a source transaction."""

    state = getattr(_CHECKPOINT_PERSISTENCE_SCOPE, "state", None)
    if not isinstance(state, tuple) or len(state) != 3:
        return False
    checkpoint_root, staged_root, identity_sha256 = state
    try:
        current_root = Path(project_root).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if current_root != staged_root:
        return False
    _persist_generation_checkpoint(
        checkpoint_root,
        staged_root,
        identity_sha256=identity_sha256,
    )
    return True


def _stage_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _STAGE_IGNORED_DIRS}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _checkpoint_router_scope(router: Any) -> dict[str, Any]:
    current = router
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        candidate_index = getattr(current, "_candidate_index", None)
        strategy = getattr(current, "_strategy", None)
        candidate_count = getattr(current, "_count", None)
        candidate_fields = (candidate_index, strategy, candidate_count)
        if any(value is not None for value in candidate_fields):
            if not (
                type(candidate_index) is int
                and isinstance(strategy, str)
                and strategy.strip()
                and type(candidate_count) is int
                and candidate_count >= 1
                and 0 <= candidate_index < candidate_count
            ):
                raise CustomModuleGenerationError(
                    "Candidate checkpoint identity requires valid index/count/strategy."
                )
            return {
                "mode": "candidate",
                "strategy_epoch": _CHECKPOINT_STRATEGY_EPOCH,
                "candidate_index": candidate_index,
                "candidate_count": candidate_count,
                "strategy": strategy.strip(),
            }
        current = getattr(current, "_router", None)
    return {"mode": "single", "strategy_epoch": _CHECKPOINT_STRATEGY_EPOCH}


def _stage_tree_snapshot(root: Path) -> tuple[str, dict[str, str]]:
    """Hash checkpoint structure and file contents in one filesystem traversal."""

    rows: list[tuple[str, str, str]] = []
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _STAGE_IGNORED_DIRS for part in relative.parts):
            continue
        normalized = relative.as_posix()
        if path.is_symlink():
            rows.append((normalized, "symlink", str(path.readlink())))
        elif path.is_file():
            digest = "sha256:" + content_digest(path).hex()
            rows.append((normalized, "file", digest))
            files[normalized] = digest
        elif path.is_dir():
            rows.append((normalized, "directory", ""))
    return _sha256_json(rows), files


def _checkpoint_tree_state_sha256(root: Path) -> str:
    return _stage_tree_snapshot(root)[0]


def _project_snapshot(root: Path) -> dict[str, str]:
    return _stage_tree_snapshot(root)[1]


def _mutable_stage_state_sha256(staged_root: Path) -> str:
    snapshot = {
        path: digest
        for path, digest in _project_snapshot(staged_root).items()
        if custom_module_path_allowed(path) and not custom_module_path_protected(path)
    }
    return _sha256_json(snapshot)


def _generation_checkpoint_identity(
    *,
    module_query: str,
    minecraft_version: str,
    loader: str,
    mappings: str,
    research_context: Any,
    router: Any,
) -> str:
    return _sha256_json(
        {
            "schema_version": _CHECKPOINT_SCHEMA,
            "module_query_sha256": _sha256_json(module_query),
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
                "mappings": mappings,
            },
            "research_context_sha256": _sha256_json(research_context),
            "router_scope": _checkpoint_router_scope(router),
        }
    )


def _checkpoint_key(identity_sha256: str) -> str:
    prefix, separator, digest = str(identity_sha256).partition(":")
    if prefix != "sha256" or separator != ":" or not _CHECKPOINT_KEY.fullmatch(digest):
        raise ValueError("Custom-module checkpoint identity must be a SHA-256 receipt")
    return digest


def _checkpoint_manifest(checkpoint_root: Path) -> Path:
    return checkpoint_root / "checkpoint.json"


def _checkpoint_base(checkpoint_root: Path) -> Path:
    return checkpoint_root / "base"


def _checkpoint_directory(root: Path, configured_root: Path | None = None) -> Path:
    base = configured_root if configured_root is not None else root.parent / _CHECKPOINT_DIRECTORY
    if base.parent.exists() and base.parent.is_symlink():
        raise CustomModuleGenerationError("Custom-module checkpoint parent may not be a symlink.")
    if base.exists() and (base.is_symlink() or not base.is_dir()):
        raise CustomModuleGenerationError(
            "Custom-module checkpoint root must be a regular host directory."
        )
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def _safe_checkpoint_path(base: Path, key: str) -> Path:
    if not _CHECKPOINT_KEY.fullmatch(key):
        raise CustomModuleGenerationError("Unsafe custom-module checkpoint key")
    checkpoint_root = base / key
    try:
        checkpoint_root.resolve().relative_to(base)
    except ValueError as exc:
        raise CustomModuleGenerationError(
            "Custom-module checkpoint escaped its host-owned root."
        ) from exc
    return checkpoint_root


def _remove_generation_checkpoint(
    checkpoint_root: Path,
    *,
    owned_lease: _GenerationCheckpointLease | None = None,
) -> None:
    declared_base = checkpoint_root.parent
    if declared_base.is_symlink() or not declared_base.is_dir():
        raise CustomModuleGenerationError(
            "Refusing to remove checkpoint through unsafe host root."
        )
    base = declared_base.resolve()
    lease_owned = bool(
        owned_lease is not None
        and owned_lease.checkpoint_root == checkpoint_root.resolve()
    )
    if base != declared_base or (
        base.name != _CHECKPOINT_DIRECTORY and not lease_owned
    ):
        raise CustomModuleGenerationError(
            "Refusing to remove unrecognized checkpoint path."
        )
    if not _CHECKPOINT_KEY.fullmatch(checkpoint_root.name) or checkpoint_root.is_symlink():
        raise CustomModuleGenerationError("Refusing to remove unsafe checkpoint path.")
    if checkpoint_root.exists():
        checkpoint_root.resolve().relative_to(base)
        shutil.rmtree(checkpoint_root)
    try:
        base.rmdir()
    except OSError:
        pass


def _persist_generation_checkpoint(
    checkpoint_root: Path,
    staged_root: Path,
    *,
    identity_sha256: str,
    stage_tree_sha256: str | None = None,
) -> None:
    checkpoint_stat = checkpoint_root.lstat()
    staged_stat = staged_root.lstat()
    base_root = _checkpoint_base(checkpoint_root)
    base_stat = base_root.lstat()
    if not all(
        stat.S_ISDIR(item.st_mode)
        for item in (checkpoint_stat, staged_stat, base_stat)
    ):
        raise ValueError("Custom-module checkpoint staging root is unsafe")
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "identity_sha256": identity_sha256,
        "base_tree_sha256": _checkpoint_tree_state_sha256(base_root),
        "stage_tree_sha256": (
            stage_tree_sha256
            if stage_tree_sha256 is not None
            else _checkpoint_tree_state_sha256(staged_root)
        ),
    }
    manifest = _checkpoint_manifest(checkpoint_root)
    if manifest.is_symlink():
        raise ValueError("Custom-module checkpoint manifest may not be a symlink")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint-",
        suffix=".tmp",
        dir=checkpoint_root,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if checkpoint_root.is_symlink() or manifest.is_symlink():
            raise ValueError("Custom-module checkpoint changed during persistence")
        os.replace(temporary_name, manifest)
        temporary_name = ""
        try:
            directory_fd = os.open(checkpoint_root, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _read_generation_checkpoint_manifest(checkpoint_root: Path) -> dict[str, Any]:
    manifest = _checkpoint_manifest(checkpoint_root)
    if manifest.is_symlink():
        raise ValueError("Custom-module checkpoint manifest may not be a symlink")
    with manifest.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Custom-module checkpoint manifest must be an object")
    return raw


def _initialize_generation_checkpoint(
    root: Path,
    checkpoint_root: Path,
    *,
    identity_sha256: str,
) -> Path:
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=False, exist_ok=False)
    base_root = _checkpoint_base(checkpoint_root)
    staged_root = checkpoint_root / "project"
    try:
        shutil.copytree(
            root,
            base_root,
            symlinks=True,
            ignore=_stage_ignore,
            copy_function=reflink_or_copy,
        )
        shutil.copytree(
            base_root,
            staged_root,
            symlinks=True,
            copy_function=reflink_or_copy,
        )
        _persist_generation_checkpoint(
            checkpoint_root,
            staged_root,
            identity_sha256=identity_sha256,
        )
    except BaseException:
        if checkpoint_root.exists() and not checkpoint_root.is_symlink():
            shutil.rmtree(checkpoint_root, ignore_errors=True)
        raise
    return staged_root


def _collect_staged_operations(
    original_root: Path,
    staged_root: Path,
    before: dict[str, str],
    *,
    after: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    del original_root
    if after is None:
        after = _project_snapshot(staged_root)
    operations: list[dict[str, Any]] = []
    touched: list[str] = []
    discarded: list[str] = []
    for path in sorted(set(before) | set(after)):
        old_sha = before.get(path)
        new_sha = after.get(path)
        if old_sha == new_sha:
            continue
        if not custom_module_path_allowed(path) or custom_module_path_protected(path):
            discarded.append(path)
            continue
        if new_sha is None:
            raise CustomModuleGenerationError(
                f"Custom-module coding agent may not delete source/resource files: {path}"
            )
        target = staged_root / Path(path)
        if target.is_symlink() or not target.is_file():
            raise CustomModuleGenerationError(
                f"Custom-module staged target is not a regular file: {path}"
            )
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CustomModuleGenerationError(
                f"Custom-module generated file must be UTF-8 text: {path}: {exc}"
            ) from exc
        if old_sha is None:
            operation = {"operation": "create", "path": path, "content": content}
        else:
            operation = {
                "operation": "replace",
                "path": path,
                "expected_sha256": old_sha,
                "content": content,
            }
        operations.append(operation)
        touched.append(path)
    return operations, touched, discarded


def _checkpoint_patch_operations(
    base_root: Path,
    staged_root: Path,
) -> list[dict[str, Any]]:
    operations, _touched, discarded = _collect_staged_operations(
        base_root,
        staged_root,
        _project_snapshot(base_root),
    )
    if discarded:
        raise CustomModuleGenerationError(
            "Resumable checkpoint contains out-of-scope source changes."
        )
    return operations


def _rebase_generation_checkpoint(
    root: Path,
    checkpoint_root: Path,
    *,
    identity_sha256: str,
) -> Path:
    base_root = _checkpoint_base(checkpoint_root)
    staged_root = checkpoint_root / "project"
    operations = _checkpoint_patch_operations(base_root, staged_root)
    next_base = Path(tempfile.mkdtemp(prefix=".base-rebase-", dir=checkpoint_root))
    next_stage = Path(tempfile.mkdtemp(prefix=".project-rebase-", dir=checkpoint_root))
    next_base.rmdir()
    next_stage.rmdir()
    try:
        shutil.copytree(
            root,
            next_base,
            symlinks=True,
            ignore=_stage_ignore,
            copy_function=reflink_or_copy,
        )
        shutil.copytree(
            next_base,
            next_stage,
            symlinks=True,
            copy_function=reflink_or_copy,
        )
        if operations:
            TransactionalSourcePatcher(next_stage).apply(operations)
        shutil.rmtree(base_root)
        shutil.rmtree(staged_root)
        os.replace(next_base, base_root)
        os.replace(next_stage, staged_root)
        _persist_generation_checkpoint(
            checkpoint_root,
            staged_root,
            identity_sha256=identity_sha256,
        )
        return staged_root
    finally:
        for temporary in (next_base, next_stage):
            if temporary.exists() and not temporary.is_symlink():
                shutil.rmtree(temporary, ignore_errors=True)


def _prepare_generation_checkpoint(
    root: Path,
    *,
    identity_sha256: str,
    configured_root: Path | None = None,
) -> tuple[Path, Path, bool, _GenerationCheckpointLease]:
    base = _checkpoint_directory(root, configured_root)
    checkpoint_root = _safe_checkpoint_path(base, _checkpoint_key(identity_sha256))
    lease = _GenerationCheckpointLease(checkpoint_root)
    try:
        staged_root = checkpoint_root / "project"
        if checkpoint_root.exists():
            reusable = False
            try:
                raw = _read_generation_checkpoint_manifest(checkpoint_root)
                base_root = _checkpoint_base(checkpoint_root)
                reusable = (
                    raw.get("schema_version") == _CHECKPOINT_SCHEMA
                    and raw.get("identity_sha256") == identity_sha256
                    and base_root.is_dir()
                    and not base_root.is_symlink()
                    and staged_root.is_dir()
                    and not staged_root.is_symlink()
                    and raw.get("base_tree_sha256")
                    == _checkpoint_tree_state_sha256(base_root)
                    and raw.get("stage_tree_sha256")
                    == _checkpoint_tree_state_sha256(staged_root)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                reusable = False
            if (
                reusable
                and raw.get("base_tree_sha256")
                != _checkpoint_tree_state_sha256(root)
            ):
                try:
                    staged_root = _rebase_generation_checkpoint(
                        root,
                        checkpoint_root,
                        identity_sha256=identity_sha256,
                    )
                except (
                    CustomModuleGenerationError,
                    OSError,
                    SourcePatchError,
                    ValueError,
                ):
                    reusable = False
            if reusable:
                return checkpoint_root, staged_root, True, lease
            _remove_generation_checkpoint(checkpoint_root)
        staged_root = _initialize_generation_checkpoint(
            root,
            checkpoint_root,
            identity_sha256=identity_sha256,
        )
    except BaseException:
        lease.close()
        raise
    return checkpoint_root, staged_root, False, lease


def _committed_patch_receipt_matches(
    result: Any,
    *,
    project_root: Path,
) -> bool:
    if not isinstance(result, Mapping):
        return False
    patch = result.get("patch_receipt")
    if not isinstance(patch, Mapping) or patch.get("status") not in {"APPLIED", "UNCHANGED"}:
        return False
    operations = patch.get("operations")
    if not isinstance(operations, list) or not operations:
        return False
    root = project_root.expanduser().resolve()
    for item in operations:
        if not isinstance(item, Mapping):
            return False
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return False
        candidate = (root / raw_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        expected = item.get("after_sha256")
        if expected is None:
            if candidate.exists():
                return False
            continue
        if (
            not isinstance(expected, str)
            or not candidate.is_file()
            or candidate.is_symlink()
            or ("sha256:" + content_digest(candidate).hex()) != expected
        ):
            return False
    return True


def finalize_persisted_generation_checkpoint(
    result: Any,
    *,
    project_root: str | Path,
    checkpoint_root: str | Path | None = None,
) -> bool:
    """Recover cleanup after ledger commit survived but in-memory ownership did not."""

    if not isinstance(result, dict):
        return False
    checkpoint = result.get("generation_checkpoint")
    if not isinstance(checkpoint, dict):
        return True
    if checkpoint.get("status") == "CLEANED_AFTER_LIVE_COMMIT":
        return True
    if (
        checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "AWAITING_LIVE_COMMIT"
    ):
        return False
    identity = checkpoint.get("identity_sha256")
    if not isinstance(identity, str):
        return False
    root = Path(project_root).expanduser().resolve()
    configured = (
        Path(checkpoint_root).expanduser()
        if checkpoint_root is not None
        else None
    )
    base = _checkpoint_directory(root, configured)
    path = _safe_checkpoint_path(base, _checkpoint_key(identity))
    resolved = path.resolve()
    with _CHECKPOINT_ACTIVE_LOCK:
        if resolved in _CHECKPOINT_ACTIVE_PATHS:
            return False

    if path.exists():
        if not _committed_patch_receipt_matches(result, project_root=root):
            return False
        try:
            manifest = _read_generation_checkpoint_manifest(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if (
            manifest.get("schema_version") != _CHECKPOINT_SCHEMA
            or manifest.get("identity_sha256") != identity
        ):
            return False
        _remove_generation_checkpoint(path)
    checkpoint["status"] = "CLEANED_AFTER_LIVE_COMMIT"
    checkpoint.pop("cleanup_token", None)
    return True
