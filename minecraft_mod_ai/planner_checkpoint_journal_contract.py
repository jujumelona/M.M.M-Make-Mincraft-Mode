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

_SAVED_DIGEST_VERSION = 1
_SAVED_DIGEST_DOMAIN = b"mmm/planner/saved-batches/v1\0"
_HOST_RESUME_DOMAIN = b"mmm/planner/host-resume/v2\0"
_HOT_STATE = threading.local()


@dataclass
class _JournalRuntime:
    queue: list[Any]
    accepted: list[Any]
    pending_remaining: int | None
    queue_signature: tuple[int, int] | None
    events_signature: tuple[int, int] | None


@dataclass
class _SavedBatchTracker:
    owner: list[Any]
    digest: str
    identities: set[str]
    accepted_ids: list[str]


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


def _valid_saved_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _saved_digest_seed() -> str:
    return hashlib.sha256(_SAVED_DIGEST_DOMAIN).hexdigest()


def _extend_saved_digest(previous: str, batch: Any, fingerprint_fn: Any) -> str:
    """Append one immutable batch to the persisted SHA256 chain in O(batch size)."""

    leaf = fingerprint_fn(batch)
    digest = hashlib.sha256()
    digest.update(_SAVED_DIGEST_DOMAIN)
    digest.update(bytes.fromhex(previous))
    digest.update(bytes.fromhex(leaf))
    return digest.hexdigest()


def _remember_loaded_saved(state: dict[str, Any]) -> None:
    saved = state.get("saved_batches")
    if not isinstance(saved, list):
        _HOT_STATE.loaded_saved_hint = None
        return
    digest = state.get("saved_batches_digest")
    count = state.get("saved_batches_digest_count")
    version = state.get("saved_batches_digest_version")
    if (
        version != _SAVED_DIGEST_VERSION
        or type(count) is not int
        or count != len(saved)
        or not _valid_saved_digest(digest)
    ):
        _HOT_STATE.loaded_saved_hint = None
        return
    _HOT_STATE.loaded_saved_hint = (
        count,
        saved[0] if saved else None,
        saved[-1] if saved else None,
        digest,
    )


def install(incremental_module: Any) -> None:
    """Install append-only planner persistence and hot-path digest reuse.

    The pending queue is written once. Each accepted batch is one fsynced append event.
    Only small mutable metadata (status, cursor, one pending patch) is atomically
    replaced. Legacy monolithic checkpoints remain readable and migrate on the next
    save.

    Durable queue/event files are replayed once per process/path and thereafter tracked
    in memory only after fsync succeeds. Large immutable requests are canonicalized once
    per planner invocation. Accepted batches use a persisted append-only SHA256 chain,
    so adding one batch hashes only that batch instead of serializing the full accepted
    prefix again. This removes the quadratic request/saved_batches fingerprint path while
    retaining deterministic resume identifiers and crash recovery.
    """

    current_save = incremental_module._save_checkpoint
    current_load = incremental_module._load_checkpoint
    if getattr(current_save, "_mmm_linear_checkpoint_journal", False):
        return

    current_fingerprint = incremental_module._fingerprint
    current_batch_identity = incremental_module._batch_identity
    current_accepted_batch_ids = incremental_module._accepted_batch_ids
    current_merge_saved_batches = incremental_module._merge_saved_batches
    current_checkpoint_path = incremental_module._checkpoint_path
    checkpoint_version = int(incremental_module._CHECKPOINT_VERSION)

    def _new_saved_tracker(
        saved: list[Any],
        *,
        restored_digest: str | None = None,
    ) -> _SavedBatchTracker:
        identities: set[str] = set()
        accepted_ids: list[str] = []
        digest = restored_digest if _valid_saved_digest(restored_digest) else _saved_digest_seed()
        rebuild_digest = restored_digest is None or not _valid_saved_digest(restored_digest)
        for value in saved:
            identities.add(current_batch_identity(value))
            if isinstance(value, dict):
                batch_id = str(value.get("batch_id", "")).strip()
                if batch_id:
                    accepted_ids.append(batch_id)
            if rebuild_digest:
                digest = _extend_saved_digest(digest, value, current_fingerprint)
        return _SavedBatchTracker(
            owner=saved,
            digest=digest,
            identities=identities,
            accepted_ids=accepted_ids,
        )

    def _tracker_for(saved: list[Any]) -> _SavedBatchTracker:
        tracker = getattr(_HOT_STATE, "saved_tracker", None)
        if isinstance(tracker, _SavedBatchTracker) and tracker.owner is saved:
            return tracker

        restored_digest: str | None = None
        hint = getattr(_HOT_STATE, "loaded_saved_hint", None)
        if isinstance(hint, tuple) and len(hint) == 4:
            count, first, last, digest = hint
            if (
                count == len(saved)
                and (not saved or (saved[0] is first and saved[-1] is last))
                and _valid_saved_digest(digest)
            ):
                restored_digest = digest

        tracker = _new_saved_tracker(saved, restored_digest=restored_digest)
        _HOT_STATE.saved_tracker = tracker
        return tracker

    def fingerprint(value: Any) -> str:
        request_obj = getattr(_HOT_STATE, "request_obj", None)
        if value is request_obj:
            request_digest = getattr(_HOT_STATE, "request_digest", None)
            if _valid_saved_digest(request_digest):
                return request_digest

        if isinstance(value, list):
            tracker = getattr(_HOT_STATE, "saved_tracker", None)
            if isinstance(tracker, _SavedBatchTracker) and tracker.owner is value:
                return tracker.digest

        # This shape is internal to host_resume_* generation. The cursor is opaque, so
        # bind it to the persisted saved-prefix digest plus only the new page digest.
        # Do not serialize/hash the entire accepted prefix again.
        if (
            isinstance(value, dict)
            and frozenset(value) == {"saved", "new"}
            and isinstance(value.get("saved"), list)
        ):
            saved = value["saved"]
            tracker = _tracker_for(saved)
            new_digest = current_fingerprint(value.get("new"))
            digest = hashlib.sha256()
            digest.update(_HOST_RESUME_DOMAIN)
            digest.update(bytes.fromhex(tracker.digest))
            digest.update(bytes.fromhex(new_digest))
            return digest.hexdigest()

        return current_fingerprint(value)

    def accepted_batch_ids(saved_batches: Any) -> list[str]:
        if not isinstance(saved_batches, list):
            return current_accepted_batch_ids(saved_batches)
        tracker = _tracker_for(saved_batches)
        return list(tracker.accepted_ids)

    def merge_saved_batches(saved: list[Any], incoming: Any) -> None:
        tracker = _tracker_for(saved)
        for value in incoming:
            identity = current_batch_identity(value)
            if identity in tracker.identities:
                continue
            saved.append(value)
            tracker.identities.add(identity)
            if isinstance(value, dict):
                batch_id = str(value.get("batch_id", "")).strip()
                if batch_id:
                    tracker.accepted_ids.append(batch_id)
            tracker.digest = _extend_saved_digest(
                tracker.digest,
                value,
                current_fingerprint,
            )

    def checkpoint_path(stage: str, request: Any) -> Path:
        # Preserve the exact legacy path digest without serializing request twice.
        # sort_keys=True orders this two-key object as request,stage.
        request_bytes = incremental_module._canonical_bytes(request)
        request_digest = hashlib.sha256(request_bytes).hexdigest()
        payload = (
            b'{"request":'
            + request_bytes
            + b',"stage":'
            + incremental_module._canonical_bytes(stage)
            + b"}"
        )
        digest = hashlib.sha256(payload).hexdigest()
        safe_stage = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in stage
        ).strip("_")[:60] or "planner"
        _HOT_STATE.request_obj = request
        _HOT_STATE.request_digest = request_digest
        return incremental_module._checkpoint_root() / f"{safe_stage}-{digest[:20]}.json"

    def load_checkpoint(path: Path) -> dict[str, Any]:
        with _path_lock(path):
            meta = _read_meta(path, checkpoint_version=checkpoint_version)
            journaled = (
                meta.get("journal_version") == _JOURNAL_VERSION
                or _queue_path(path).is_file()
                or _events_path(path).is_file()
            )
            if not journaled:
                state = current_load(path)
                if isinstance(state, dict):
                    _remember_loaded_saved(state)
                return state

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
            _remember_loaded_saved(state)
            return state

    def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
        with _path_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            saved_source = state.get("saved_batches", [])
            if isinstance(saved_source, list):
                tracker = _tracker_for(saved_source)
                state["saved_batches_digest_version"] = _SAVED_DIGEST_VERSION
                state["saved_batches_digest_count"] = len(saved_source)
                state["saved_batches_digest"] = tracker.digest
            saved = list(saved_source)
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

    fingerprint._mmm_incremental_fingerprint_cache = True  # type: ignore[attr-defined]
    accepted_batch_ids._mmm_incremental_identity_cache = True  # type: ignore[attr-defined]
    merge_saved_batches._mmm_incremental_identity_cache = True  # type: ignore[attr-defined]
    checkpoint_path._mmm_request_fingerprint_cache = True  # type: ignore[attr-defined]
    checkpoint_path.__wrapped__ = current_checkpoint_path  # type: ignore[attr-defined]
    merge_saved_batches.__wrapped__ = current_merge_saved_batches  # type: ignore[attr-defined]

    save_checkpoint._mmm_linear_checkpoint_journal = True  # type: ignore[attr-defined]
    load_checkpoint._mmm_linear_checkpoint_journal = True  # type: ignore[attr-defined]
    save_checkpoint._mmm_cached_journal_replay = True  # type: ignore[attr-defined]
    load_checkpoint._mmm_cached_journal_replay = True  # type: ignore[attr-defined]
    save_checkpoint._mmm_saved_digest_chain = True  # type: ignore[attr-defined]
    load_checkpoint._mmm_saved_digest_chain = True  # type: ignore[attr-defined]
    save_checkpoint.__wrapped__ = current_save  # type: ignore[attr-defined]
    load_checkpoint.__wrapped__ = current_load  # type: ignore[attr-defined]

    incremental_module._fingerprint = fingerprint
    incremental_module._accepted_batch_ids = accepted_batch_ids
    incremental_module._merge_saved_batches = merge_saved_batches
    incremental_module._checkpoint_path = checkpoint_path
    incremental_module._save_checkpoint = save_checkpoint
    incremental_module._load_checkpoint = load_checkpoint


__all__ = ["install"]