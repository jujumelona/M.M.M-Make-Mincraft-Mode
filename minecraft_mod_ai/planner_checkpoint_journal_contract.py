from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


_JOURNAL_VERSION = 1


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


def install(incremental_module: Any) -> None:
    """Replace quadratic full-state rewrites with append-only planner journals.

    The pending queue is written once. Each accepted batch is one fsynced append event.
    Only small mutable metadata (status, cursor, one pending patch) is atomically
    replaced. Legacy monolithic checkpoints remain readable and migrate on the next
    save.

    Recovery authority is deliberately ordered as queue -> accepted events -> metadata.
    A crash may happen after the immutable queue is atomically installed but before the
    small metadata file is replaced. In that window stale metadata must never be allowed
    to claim that the newly durable queue has zero remaining work.
    """

    current_save = incremental_module._save_checkpoint
    current_load = incremental_module._load_checkpoint
    if getattr(current_save, "_mmm_linear_checkpoint_journal", False):
        return

    checkpoint_version = int(incremental_module._CHECKPOINT_VERSION)

    def load_checkpoint(path: Path) -> dict[str, Any]:
        meta = _read_meta(path, checkpoint_version=checkpoint_version)
        journaled = (
            meta.get("journal_version") == _JOURNAL_VERSION
            or _queue_path(path).is_file()
            or _events_path(path).is_file()
        )
        if not journaled:
            return current_load(path)

        accepted, event_pending_remaining = _replay(path)
        queue = _read_jsonl(_queue_path(path))

        # The immutable queue is the authoritative record that work exists. Once at
        # least one fsynced accept event exists, its pending_remaining is the durable
        # progress cursor. Without an event, ALL queue entries remain pending even if
        # an older metadata file happens to say pending_remaining=0.
        remaining_count = (
            event_pending_remaining
            if event_pending_remaining is not None
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
        queue = _read_jsonl(_queue_path(path))

        accepted_existing, _ = _replay(path)
        existing_count = len(accepted_existing)
        if legacy_state and not accepted_existing:
            legacy_saved = list(legacy_state.get("saved_batches", []))
            for index, batch in enumerate(legacy_saved):
                _append_event(
                    path,
                    {
                        "type": "accept",
                        "accepted_index": index,
                        "event_id": _event_id(index, batch),
                        "batch": batch,
                        "pending_remaining": len(queue),
                    },
                )
            existing_count = len(legacy_saved)

        # Append only the newly accepted suffix. accepted_index makes a repeated append
        # after a crash idempotent when replayed.
        pending_remaining = len(pending) if queue else 0
        start = min(existing_count, len(saved))
        for index in range(start, len(saved)):
            batch = saved[index]
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
    save_checkpoint.__wrapped__ = current_save  # type: ignore[attr-defined]
    load_checkpoint.__wrapped__ = current_load  # type: ignore[attr-defined]
    incremental_module._save_checkpoint = save_checkpoint
    incremental_module._load_checkpoint = load_checkpoint


__all__ = ["install"]
