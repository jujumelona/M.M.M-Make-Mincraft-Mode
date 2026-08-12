from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_JOURNAL_VERSION = 1
_RUNTIME_CACHE_LIMIT = 32
_RUNTIME_CACHE_LOCK = threading.RLock()
_RUNTIME_CACHE: dict[str, "_JournalRuntime"] = {}
_PATH_LOCKS: dict[str, threading.RLock] = {}


@dataclass
class _JournalRuntime:
    queue: list[Any]
    accepted: list[Any]
    pending_remaining: int | None
    queue_signature: tuple[int, int] | None
    events_signature: tuple[int, int] | None


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def _queue_path(path: Path) -> Path:
    return _sidecar(path, ".pending.jsonl")


def _events_path(path: Path) -> Path:
    return _sidecar(path, ".events.jsonl")


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns)


def _cache_key(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _path_lock(path: Path) -> threading.RLock:
    key = _cache_key(path)
    with _RUNTIME_CACHE_LOCK:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
            while len(_PATH_LOCKS) > _RUNTIME_CACHE_LIMIT:
                old_key = next(iter(_PATH_LOCKS))
                if old_key == key:
                    break
                _PATH_LOCKS.pop(old_key, None)
        return lock


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(_compact(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_queue_once(path: Path, pending: list[Any]) -> None:
    queue = _queue_path(path)
    if queue.is_file():
        return
    tmp = queue.with_suffix(queue.suffix + ".tmp")
    queue.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        for value in pending:
            handle.write(_compact(value))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, queue)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    events = _events_path(path)
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(_compact(event))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    result: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # A final torn append cannot invalidate earlier fsynced records.
                break
    return result


def _read_meta(path: Path, *, checkpoint_version: int) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(value, dict) or value.get("version") != checkpoint_version:
        return {}
    return value


def _event_id(index: int, batch: Any) -> str:
    payload = f"{index}:" + _compact(batch)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replay(path: Path) -> tuple[list[Any], int | None]:
    """Replay idempotent accept events; later duplicate indices do not duplicate work."""

    accepted_by_index: dict[int, Any] = {}
    pending_remaining: int | None = None
    for raw in _read_jsonl(_events_path(path)):
        if not isinstance(raw, dict) or raw.get("type") != "accept":
            continue
        index = raw.get("accepted_index")
        if type(index) is not int or index < 0:
            continue
        batch = raw.get("batch")
        event_id = str(raw.get("event_id", ""))
        if event_id != _event_id(index, batch):
            continue
        accepted_by_index[index] = batch
        remaining = raw.get("pending_remaining")
        if type(remaining) is int and remaining >= 0:
            pending_remaining = (
                remaining
                if pending_remaining is None
                else min(pending_remaining, remaining)
            )

    accepted = [accepted_by_index[index] for index in sorted(accepted_by_index)]
    return accepted, pending_remaining


def _hydrate_runtime(path: Path) -> _JournalRuntime:
    """Replay durable sidecars once, then reuse them until either file changes."""

    key = _cache_key(path)
    queue_path = _queue_path(path)
    events_path = _events_path(path)
    queue_signature = _file_signature(queue_path)
    events_signature = _file_signature(events_path)
    with _RUNTIME_CACHE_LOCK:
        cached = _RUNTIME_CACHE.get(key)
        if (
            cached is not None
            and cached.queue_signature == queue_signature
            and cached.events_signature == events_signature
        ):
            return cached

    queue = _read_jsonl(queue_path)
    accepted, pending_remaining = _replay(path)
    runtime = _JournalRuntime(
        queue=queue,
        accepted=accepted,
        pending_remaining=pending_remaining,
        queue_signature=queue_signature,
        events_signature=events_signature,
    )
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE.pop(key, None)
        _RUNTIME_CACHE[key] = runtime
        while len(_RUNTIME_CACHE) > _RUNTIME_CACHE_LIMIT:
            _RUNTIME_CACHE.pop(next(iter(_RUNTIME_CACHE)))
    return runtime


def _refresh_runtime_signatures(path: Path, runtime: _JournalRuntime) -> None:
    runtime.queue_signature = _file_signature(_queue_path(path))
    runtime.events_signature = _file_signature(_events_path(path))


def _record_accepted(
    path: Path,
    runtime: _JournalRuntime,
    *,
    index: int,
    batch: Any,
    pending_remaining: int,
) -> None:
    """Append durably first, then advance the same-process replay cache."""

    _append_event(
        path,
        {
            "type": "accept",
            "accepted_index": index,
            "event_id": _event_id(index, batch),
            "batch": batch,
            "pending_remaining": pending_remaining,
        },
    )
    if index < len(runtime.accepted):
        runtime.accepted[index] = batch
    elif index == len(runtime.accepted):
        runtime.accepted.append(batch)
    else:
        # Normal planner writes are contiguous. If a caller ever produces a hole,
        # invalidate and replay rather than inventing missing accepted work.
        with _RUNTIME_CACHE_LOCK:
            _RUNTIME_CACHE.pop(_cache_key(path), None)
        return
    runtime.pending_remaining = (
        pending_remaining
        if runtime.pending_remaining is None
        else min(runtime.pending_remaining, pending_remaining)
    )
    runtime.events_signature = _file_signature(_events_path(path))


def install(incremental_module: Any) -> None:
    """Replace quadratic full-state rewrites with append-only planner journals.

    The pending queue is written once. Each accepted batch is one fsynced append event.
    Only small mutable metadata (status, cursor, one pending patch) is atomically
    replaced. Legacy monolithic checkpoints remain readable and migrate on the next
    save.

    The durable queue/event files are replayed once per process/path and thereafter
    tracked in memory only after fsync succeeds. A fresh process or an externally
    changed sidecar automatically replays the disk journal, preserving crash recovery
    while avoiding O(N^2) read/parse work during long paged generations.
    """

    current_save = incremental_module._save_checkpoint
    current_load = incremental_module._load_checkpoint
    if getattr(current_save, "_mmm_linear_checkpoint_journal", False):
        return

    checkpoint_version = int(incremental_module._CHECKPOINT_VERSION)

    def load_checkpoint(path: Path) -> dict[str, Any]:
        with _path_lock(path):
            meta = _read_meta(path, checkpoint_version=checkpoint_version)
            journaled = (
                meta.get("journal_version") == _JOURNAL_VERSION
                or _queue_path(path).is_file()
                or _events_path(path).is_file()
            )
            if not journaled:
                return current_load(path)

            runtime = _hydrate_runtime(path)
            accepted = list(runtime.accepted)
            queue = runtime.queue

            # The immutable queue is the authoritative record that work exists. Once
            # at least one fsynced accept event exists, its pending_remaining is the
            # durable progress cursor. Without an event, ALL queue entries remain
            # pending even if stale metadata says pending_remaining=0.
            remaining_count = (
                runtime.pending_remaining
                if runtime.pending_remaining is not None
                else len(queue)
            )
            remaining_count = max(0, min(int(remaining_count), len(queue)))
            pending = queue[len(queue) - remaining_count :] if remaining_count else []

            state = {
                key: value
                for key, value in meta.items()
                if key
                not in {
                    "version",
                    "journal_version",
                    "accepted_count",
                    "pending_count",
                    "pending_remaining",
                }
            }
            state["saved_batches"] = accepted
            state["pending_batches"] = pending
            return state

    def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
        with _path_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = list(state.get("saved_batches", []))
            pending = list(state.get("pending_batches", []))

            old_meta = _read_meta(path, checkpoint_version=checkpoint_version)
            old_journal = old_meta.get("journal_version") == _JOURNAL_VERSION
            legacy_state = current_load(path) if path.is_file() and not old_journal else {}

            # The queue begins at the first journal save that actually has pending work.
            # It is immutable afterwards, so shrinking progress never rewrites it.
            if pending:
                _write_queue_once(path, pending)
            runtime = _hydrate_runtime(path)
            queue = runtime.queue

            existing_count = len(runtime.accepted)
            if legacy_state and not runtime.accepted:
                legacy_saved = list(legacy_state.get("saved_batches", []))
                for index, batch in enumerate(legacy_saved):
                    _record_accepted(
                        path,
                        runtime,
                        index=index,
                        batch=batch,
                        pending_remaining=len(queue),
                    )
                existing_count = len(legacy_saved)

            # Append only the newly accepted suffix. accepted_index makes a repeated
            # append after a crash idempotent when replayed.
            pending_remaining = len(pending) if queue else 0
            start = min(existing_count, len(saved))
            for index in range(start, len(saved)):
                _record_accepted(
                    path,
                    runtime,
                    index=index,
                    batch=saved[index],
                    pending_remaining=pending_remaining,
                )

            # Keep signatures synchronized even if there were no new events. This is
            # cheap stat metadata, not a replay/read of the journal contents.
            _refresh_runtime_signatures(path, runtime)

            small = {
                key: value
                for key, value in state.items()
                if key not in {"saved_batches", "pending_batches"}
            }
            small.update(
                {
                    "version": checkpoint_version,
                    "journal_version": _JOURNAL_VERSION,
                    "accepted_count": len(saved),
                    "pending_count": len(queue),
                    "pending_remaining": pending_remaining,
                }
            )
            _atomic_json(path, small)

    save_checkpoint._mmm_linear_checkpoint_journal = True  # type: ignore[attr-defined]
    load_checkpoint._mmm_linear_checkpoint_journal = True  # type: ignore[attr-defined]
    save_checkpoint._mmm_cached_journal_replay = True  # type: ignore[attr-defined]
    load_checkpoint._mmm_cached_journal_replay = True  # type: ignore[attr-defined]
    save_checkpoint.__wrapped__ = current_save  # type: ignore[attr-defined]
    load_checkpoint.__wrapped__ = current_load  # type: ignore[attr-defined]
    incremental_module._save_checkpoint = save_checkpoint
    incremental_module._load_checkpoint = load_checkpoint


__all__ = ["install"]