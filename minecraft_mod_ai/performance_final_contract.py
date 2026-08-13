from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import threading
from difflib import SequenceMatcher
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable

from .project_write_lock import project_write_lock


_SHARED_WRITER_FALLBACK_LOCK = threading.RLock()
_CAPTURE = threading.local()
_FICLONE = 0x40049409
_SKIP_STAGE_SUFFIXES = {
    ".class",
    ".jar",
    ".ogg",
    ".png",
    ".wav",
    ".mp3",
    ".zip",
}
_SKIP_STAGE_DIRS = {
    ".gradle",
    "build",
    "logs",
    "run",
}
_MISSING = object()


class StagedCommitConflict(RuntimeError):
    pass


def install(
    orchestrator_module: Any,
    custom_module_generator_module: Any,
    source_patch_module: Any,
) -> None:
    """Install the final local-generation concurrency contract.

    Expensive custom LLM work executes against an isolated source snapshot and only
    its short, hash-guarded commit touches the live Fabric project. Deterministic
    generators that mutate shared registries remain in the commit lane, while image
    generation and standalone audio synthesis can execute concurrently because their
    project outputs are disjoint.
    """

    _install_locked_source_patcher(source_patch_module)
    _install_serial_shared_writers(orchestrator_module)
    _install_staged_custom_generator(
        custom_module_generator_module,
        source_patch_module,
    )


def _install_locked_source_patcher(source_patch_module: Any) -> None:
    patcher = source_patch_module.TransactionalSourcePatcher
    original = patcher.apply
    if getattr(original, "_mmm_path_commit_contract", False):
        return

    @wraps(original)
    def locked_apply(self: Any, operations: Iterable[dict[str, Any]]):
        operation_list = [copy.deepcopy(item) for item in operations]
        capture_records = getattr(_CAPTURE, "records", None)
        staging_root = getattr(_CAPTURE, "staging_root", None)
        root = Path(self.project_root).resolve()
        if capture_records is not None and staging_root == root:
            before: dict[str, bytes | None] = {}
            for item in operation_list:
                relative = str(item.get("path", ""))
                target = root / relative
                before[relative] = (
                    target.read_bytes()
                    if target.is_file() and not target.is_symlink()
                    else None
                )
            capture_records.append(
                {
                    "root": str(root),
                    "operations": copy.deepcopy(operation_list),
                    "before": before,
                }
            )
            return original(self, operation_list)
        with project_write_lock(root):
            return original(self, operation_list)

    locked_apply._mmm_path_commit_contract = True
    patcher.apply = locked_apply



def _project_root_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path | None:
    candidates: list[Any] = []
    for key in ("project_root", "root", "workspace_root"):
        if key in kwargs:
            candidates.append(kwargs[key])
    candidates.extend(args)
    for value in candidates:
        if not isinstance(value, (str, Path)):
            continue
        try:
            candidate = Path(value).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    return None

def _install_serial_shared_writers(orchestrator_module: Any) -> None:
    # These functions read-modify-write shared Java, JSON, language, registry or
    # manifest files. Keep only this short mutation class serialized. Expensive
    # image generation and standalone OGG synthesis intentionally stay outside it.
    for name in (
        "generate_extended_content",
        "generate_system_pack",
        "generate_geckolib_entity_assets",
        "generate_local_ai_sidecar",
        "write_research_shard",
        "finalize_audio_registry",
        "generate_audio_assets",
    ):
        original = getattr(orchestrator_module, name, None)
        if not callable(original) or getattr(original, "_mmm_shared_writer", False):
            continue

        @wraps(original)
        def serialized(*args: Any, __original: Callable[..., Any] = original, **kwargs: Any):
            project_root = _project_root_from_call(args, kwargs)
            if project_root is None:
                # Keep safety for an unusual writer signature we cannot bind to a
                # project, without forcing normal independent projects through this lock.
                with _SHARED_WRITER_FALLBACK_LOCK:
                    return __original(*args, **kwargs)
            with project_write_lock(project_root):
                return __original(*args, **kwargs)

        serialized._mmm_shared_writer = True
        setattr(orchestrator_module, name, serialized)


def _install_staged_custom_generator(
    custom_module_generator_module: Any,
    source_patch_module: Any,
) -> None:
    cls = custom_module_generator_module.CustomModuleGenerator
    original = cls.generate
    if getattr(original, "_mmm_staged_custom", False):
        return

    @wraps(original)
    def staged_generate(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        live_root = Path(project_root).expanduser().resolve()
        if not live_root.is_dir() or live_root.is_symlink():
            return original(self, project_root, *args, **kwargs)

        # A consistent source snapshot is needed only while the cheap clone is made;
        # the long LLM call then runs without holding the project mutation lock.
        with project_write_lock(live_root):
            staging_root = _clone_source_snapshot(live_root)

        previous_index = getattr(self, "_cached_index", None)
        previous_root = getattr(self, "_cached_root", None)
        records: list[dict[str, Any]] = []
        old_records = getattr(_CAPTURE, "records", None)
        old_staging_root = getattr(_CAPTURE, "staging_root", None)
        _CAPTURE.records = records
        _CAPTURE.staging_root = staging_root
        try:
            result = original(self, staging_root, *args, **kwargs)
            if not isinstance(result, dict):
                raise StagedCommitConflict(
                    "Custom generator returned a non-object staged result."
                )
            captured = _select_custom_patch_capture(records, result)
            with project_write_lock(live_root):
                commit_receipt = _commit_staged_operations(
                    live_root=live_root,
                    staging_root=staging_root,
                    capture=captured,
                    source_patch_module=source_patch_module,
                )
            rewritten = _rewrite_root_paths(result, staging_root, live_root)
            rewritten["patch_receipt"] = commit_receipt
            rewritten["staging_receipt"] = {
                "schema_version": "mmm/custom-staging-v1",
                "status": "COMMITTED",
                "isolated_generation": True,
                "operation_count": len(captured["operations"]),
                "live_project_root": str(live_root),
            }
            return rewritten
        finally:
            if old_records is None:
                try:
                    delattr(_CAPTURE, "records")
                except AttributeError:
                    pass
            else:
                _CAPTURE.records = old_records
            if old_staging_root is None:
                try:
                    delattr(_CAPTURE, "staging_root")
                except AttributeError:
                    pass
            else:
                _CAPTURE.staging_root = old_staging_root
            self._cached_index = previous_index
            self._cached_root = previous_root
            shutil.rmtree(staging_root, ignore_errors=True)

    staged_generate._mmm_staged_custom = True
    cls.generate = staged_generate


def _clone_source_snapshot(live_root: Path) -> Path:
    parent = live_root.parent / ".mmm-parallel-staging"
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="custom-", dir=parent)).resolve()
    # mkdtemp creates the target; copytree needs a missing directory.
    stage.rmdir()

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        base = Path(directory)
        for name in names:
            path = base / name
            # copytree already enumerates every source directory. Reject links in
            # this traversal so staging safety does not require a second rglob pass.
            if path.is_symlink():
                raise StagedCommitConflict(
                    f"Staging refused project symlink: {path}"
                )
            if name in _SKIP_STAGE_DIRS and path.is_dir():
                ignored.add(name)
                continue
            if path.is_file() and path.suffix.lower() in _SKIP_STAGE_SUFFIXES:
                ignored.add(name)
        return ignored

    try:
        shutil.copytree(
            live_root,
            stage,
            copy_function=_reflink_or_copy,
            ignore=ignore,
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return stage


def _reflink_or_copy(source: str, target: str) -> str:
    if os.name == "posix":
        try:
            import fcntl

            with open(source, "rb") as src, open(target, "wb") as dst:
                fcntl.ioctl(dst.fileno(), _FICLONE, src.fileno())
            shutil.copystat(source, target)
            return target
        except (OSError, ImportError):
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
    return shutil.copy2(source, target)


def _select_custom_patch_capture(
    records: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    receipt = result.get("patch_receipt")
    receipt_ops = receipt.get("operations", []) if isinstance(receipt, dict) else []
    receipt_paths = [
        str(item.get("path", ""))
        for item in receipt_ops
        if isinstance(item, dict)
    ]
    for record in reversed(records):
        operation_paths = [
            str(item.get("path", ""))
            for item in record.get("operations", [])
            if isinstance(item, dict)
        ]
        if operation_paths == receipt_paths and operation_paths:
            return record
    if records:
        return records[-1]
    raise StagedCommitConflict(
        "Custom staging could not capture the generated patch transaction."
    )


def _commit_staged_operations(
    *,
    live_root: Path,
    staging_root: Path,
    capture: dict[str, Any],
    source_patch_module: Any,
) -> dict[str, Any]:
    operations = capture.get("operations", [])
    before = capture.get("before", {})
    if not isinstance(operations, list) or not operations:
        raise StagedCommitConflict("Staged custom patch contains no operations.")

    rebased: list[dict[str, Any]] = []
    unchanged_receipts: list[dict[str, Any]] = []
    for raw in operations:
        if not isinstance(raw, dict):
            raise StagedCommitConflict("Staged custom patch operation is invalid.")
        operation = copy.deepcopy(raw)
        relative = str(operation.get("path", ""))
        staged_path = (staging_root / relative).resolve()
        live_path = (live_root / relative).resolve()
        try:
            staged_path.relative_to(staging_root)
            live_path.relative_to(live_root)
        except ValueError as exc:
            raise StagedCommitConflict(
                f"Staged custom path escaped project root: {relative}"
            ) from exc

        kind = str(operation.get("operation", ""))
        if kind == "create":
            desired = str(operation.get("content", ""))
            if live_path.exists():
                if live_path.is_file() and live_path.read_text(encoding="utf-8") == desired:
                    unchanged_receipts.append(
                        {
                            "path": relative,
                            "operation": "create",
                            "before_sha256": source_patch_module.sha256_file(live_path),
                            "after_sha256": source_patch_module.sha256_file(live_path),
                        }
                    )
                    continue
                raise StagedCommitConflict(
                    f"Concurrent create conflict for {relative}"
                )
            rebased.append(operation)
            continue

        if not live_path.is_file() or live_path.is_symlink():
            raise StagedCommitConflict(
                f"Concurrent patch target is missing or unsafe: {relative}"
            )
        current_sha = source_patch_module.sha256_file(live_path)
        expected_sha = str(operation.get("expected_sha256", ""))
        if current_sha == expected_sha:
            rebased.append(operation)
            continue

        base_bytes = before.get(relative)
        if not isinstance(base_bytes, (bytes, bytearray)):
            raise StagedCommitConflict(
                f"No base snapshot is available to rebase {relative}"
            )
        base_text = bytes(base_bytes).decode("utf-8")
        current_text = live_path.read_text(encoding="utf-8")
        desired_text = staged_path.read_text(encoding="utf-8")
        if current_text == desired_text:
            unchanged_receipts.append(
                {
                    "path": relative,
                    "operation": kind,
                    "before_sha256": current_sha,
                    "after_sha256": current_sha,
                }
            )
            continue

        merged = _three_way_merge(
            relative,
            base_text=base_text,
            staged_text=desired_text,
            live_text=current_text,
        )
        rebased.append(
            {
                "operation": "replace",
                "path": relative,
                "expected_sha256": current_sha,
                "content": merged,
            }
        )

    if rebased:
        applied = source_patch_module.TransactionalSourcePatcher(live_root).apply(rebased)
    else:
        applied = {
            "schema_version": "mmm/source-patch-receipt-v1",
            "status": "UNCHANGED",
            "project_root": str(live_root),
            "operations": [],
        }
    if unchanged_receipts:
        applied = dict(applied)
        applied["operations"] = [
            *applied.get("operations", []),
            *unchanged_receipts,
        ]
    return applied


def _three_way_merge(
    relative: str,
    *,
    base_text: str,
    staged_text: str,
    live_text: str,
) -> str:
    if staged_text == base_text:
        return live_text
    if live_text == base_text:
        return staged_text
    if staged_text == live_text:
        return staged_text

    if relative.lower().endswith(".json"):
        try:
            base = json.loads(base_text)
            staged = json.loads(staged_text)
            live = json.loads(live_text)
            merged = _merge_json_value(base, staged, live, path=relative)
            return json.dumps(
                merged,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n"
        except (json.JSONDecodeError, StagedCommitConflict):
            pass

    return _merge_text_lines(
        relative,
        base_text=base_text,
        staged_text=staged_text,
        live_text=live_text,
    )


def _merge_json_value(base: Any, staged: Any, live: Any, *, path: str) -> Any:
    if staged == live:
        return staged
    if staged == base:
        return live
    if live == base:
        return staged
    if isinstance(base, dict) and isinstance(staged, dict) and isinstance(live, dict):
        merged: dict[str, Any] = {}
        for key in sorted(set(base) | set(staged) | set(live)):
            base_value = base.get(key, _MISSING)
            staged_value = staged.get(key, _MISSING)
            live_value = live.get(key, _MISSING)
            value = _merge_json_value(
                base_value,
                staged_value,
                live_value,
                path=f"{path}.{key}",
            )
            if value is not _MISSING:
                merged[key] = value
        return merged
    if isinstance(base, list) and isinstance(staged, list) and isinstance(live, list):
        if _is_prefix(base, staged) and _is_prefix(base, live):
            merged = list(base)
            for item in [*live[len(base):], *staged[len(base):]]:
                if item not in merged:
                    merged.append(item)
            return merged
    if base is _MISSING:
        if staged is _MISSING:
            return live
        if live is _MISSING or staged == live:
            return staged
    if staged is _MISSING and live == base:
        return _MISSING
    if live is _MISSING and staged == base:
        return _MISSING
    raise StagedCommitConflict(f"Concurrent JSON merge conflict at {path}")


def _is_prefix(base: list[Any], value: list[Any]) -> bool:
    return len(value) >= len(base) and value[: len(base)] == base


def _merge_text_lines(
    relative: str,
    *,
    base_text: str,
    staged_text: str,
    live_text: str,
) -> str:
    base = base_text.splitlines(keepends=True)
    staged = staged_text.splitlines(keepends=True)
    live = live_text.splitlines(keepends=True)
    staged_edits = _line_edits(base, staged)
    live_edits = _line_edits(base, live)
    merged_edits: list[tuple[int, int, list[str]]] = list(live_edits)

    for candidate in staged_edits:
        matched = False
        for index, existing in enumerate(merged_edits):
            if candidate[:2] == existing[:2]:
                if candidate[2] == existing[2]:
                    matched = True
                    break
                if candidate[0] == candidate[1]:
                    combined = list(existing[2])
                    for line in candidate[2]:
                        if line not in combined:
                            combined.append(line)
                    merged_edits[index] = (
                        existing[0],
                        existing[1],
                        combined,
                    )
                    matched = True
                    break
                raise StagedCommitConflict(
                    f"Concurrent text replacement conflict in {relative}"
                )
            if _edit_ranges_overlap(candidate, existing):
                raise StagedCommitConflict(
                    f"Concurrent text edit overlap in {relative}"
                )
        if not matched:
            merged_edits.append(candidate)

    result = list(base)
    for start, end, replacement in sorted(
        merged_edits,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        result[start:end] = replacement
    return "".join(result)


def _line_edits(
    base: list[str],
    variant: list[str],
) -> list[tuple[int, int, list[str]]]:
    edits: list[tuple[int, int, list[str]]] = []
    matcher = SequenceMatcher(a=base, b=variant, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            edits.append((i1, i2, variant[j1:j2]))
    return edits


def _edit_ranges_overlap(
    left: tuple[int, int, list[str]],
    right: tuple[int, int, list[str]],
) -> bool:
    l1, l2, _ = left
    r1, r2, _ = right
    if l1 == l2 and r1 == r2:
        return l1 == r1
    if l1 == l2:
        return r1 < l1 < r2
    if r1 == r2:
        return l1 < r1 < l2
    return max(l1, r1) < min(l2, r2)


def _rewrite_root_paths(value: Any, stage: Path, live: Path) -> Any:
    stage_text = str(stage)
    live_text = str(live)
    if isinstance(value, str):
        if value == stage_text:
            return live_text
        prefix = stage_text + os.sep
        if value.startswith(prefix):
            return live_text + os.sep + value[len(prefix):]
        return value
    if isinstance(value, list):
        return [_rewrite_root_paths(item, stage, live) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_root_paths(item, stage, live) for item in value)
    if isinstance(value, dict):
        return {
            key: _rewrite_root_paths(item, stage, live)
            for key, item in value.items()
        }
    return value
