"""Crash-safe, requirement-isolated behavior plans with cross-process ownership.

Each accepted requirement is committed independently. The v2 shared-map
checkpoint is read-only migration input, never a write destination. A lock is
held from checking disk through model inference and publication for the scope:
concurrent notebooks cannot independently produce or overwrite the same plan.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .model_concurrency import model_execution_publication

try:
    import fcntl  # Linux / Colab
except ImportError:  # pragma: no cover - Windows execution
    fcntl = None
    import msvcrt

_SCHEMA = "mmm/behavior-plan-checkpoint-v3"
_LEGACY_SCHEMA = "mmm/behavior-plan-checkpoint-v2"
_THREAD_GUARD = threading.Lock()
_THREAD_LOCKS = {}


def _digest(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _scope_mutex(key):
    with _THREAD_GUARD:
        if key not in _THREAD_LOCKS:
            _THREAD_LOCKS[key] = threading.RLock()
        return _THREAD_LOCKS[key]


class BehaviorPlanningCheckpoint:
    def __init__(self, scope, router):
        configured = os.environ.get("MMM_CONSTRUCTION_BEHAVIOR_CACHE_ROOT", "").strip()
        colab = bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())
        self.enabled = bool(configured or colab)
        if not self.enabled:
            return
        root = Path(configured or "/content/mmm-output/plan-checkpoints/construction-behaviors")
        try:
            role = router.registry.role(router.profile, "planner")
            model_id = str(getattr(role, "model_id", "") or "")
        except (KeyError, ValueError, AttributeError, TypeError):
            model_id = str(getattr(router, "profile", "") or "")
        context = {
            "prompt": scope.get("requested_prompt"),
            "requirements": scope.get("requirements"),
            "model_id": model_id,
        }
        self._prompt = context["prompt"]
        self._requirements = context["requirements"]
        self._model_id = context["model_id"]
        self.identity = _digest({"schema": _SCHEMA, **context})
        old_identity = _digest({"schema": _LEGACY_SCHEMA, **context})
        self.root = root / "v3" / self.identity
        self.legacy_path = root / (old_identity + ".json")
        self.lock_path = root / "v3" / (self.identity + ".lock")
        self._mutex = _scope_mutex(str(self.lock_path.resolve()))

    @contextmanager
    def claim(self, timeout_seconds=1800):
        """Claim whole-scope inference once, including across Colab kernels.

        A fixed timeout prevents indefinite cache locking. The Linux/Colab
        flock is released by the OS if the Python process crashes.
        """
        if not self.enabled:
            yield
            return
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._mutex.acquire(blocking=False):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_LOCK_TIMEOUT: thread")
            time.sleep(0.1)
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as lock_file:
                if fcntl is None:  # Windows msvcrt byte-range lock
                    lock_file.seek(0, os.SEEK_END)
                    if lock_file.tell() == 0:
                        lock_file.write(bytes([0]))
                        lock_file.flush()
                while True:
                    try:
                        if fcntl is None:
                            lock_file.seek(0)
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except (BlockingIOError, PermissionError, OSError) as exc:
                        if not isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) not in (
                            11, 13, 33, 36,
                        ):
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_LOCK_TIMEOUT: process") from exc
                        time.sleep(0.2)
                try:
                    yield
                finally:
                    if fcntl is None:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._mutex.release()

    def _path(self, requirement):
        return self.root / (
            _digest({
                "parent": requirement["requirement_id"],
                "source": requirement.get("source_span", {}).get("text")
                          or requirement["statement"],
            }) + ".json"
        )

    def _read_envelope(self, path, *, schema, identity):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CORRUPT: " + str(path)) from exc
        if not isinstance(data, dict) or data.get("schema") != schema or data.get("identity") != identity:
            raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_IDENTITY: " + str(path))
        return data

    @staticmethod
    def _verify_rows(requirement, rows):
        from .construction_task import BehaviorRequirement
        from .request_scope import _source_quote_bounds
        parent = requirement["requirement_id"]
        source = str(requirement.get("source_span", {}).get("text") or requirement["statement"])
        if not isinstance(rows, list) or not rows:
            raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_EMPTY: " + parent)
        result, identities = [], set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_INVALID_RECORD: " + parent)
            record = BehaviorRequirement(**{
                **row,
                "state_reads": tuple(row.get("state_reads", ())),
                "state_writes": tuple(row.get("state_writes", ())),
                "runtime_state_reads": tuple(row.get("runtime_state_reads", ())),
                "runtime_state_writes": tuple(row.get("runtime_state_writes", ())),
            })
            if record.requirement_id != parent or record.behavior_id in identities:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_PARENT: " + parent)
            if _source_quote_bounds(source, record.source_quote) is None:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_SOURCE_DRIFT: " + parent)
            from .construction_trigger_provenance import preserve_grounded_trigger
            cleaned, diagnostic = preserve_grounded_trigger(
                record.trigger, source_quote=record.source_quote, source_text=source)
            if diagnostic:
                # Existing checkpoints remain immutable. Repaired semantic
                # input is an execution-time view, never written back as
                # though the LLM originally generated a valid trigger.
                from dataclasses import replace
                record = replace(record, trigger=cleaned)
                print("CONSTRUCTION_TRIGGER_PROVENANCE: " + json.dumps({
                    "requirement_id": parent, "behavior_id": record.behavior_id,
                    "restored_checkpoint": True, **diagnostic,
                }, ensure_ascii=False), flush=True)
            identities.add(record.behavior_id)
            result.append(record)
        return tuple(result)

    def load(self, requirement):
        if not self.enabled:
            return None
        path = self._path(requirement)
        if path.exists():
            data = self._read_envelope(path, schema=_SCHEMA, identity=self.identity)
            rows = data.get("rows")
            if data.get("digest") != _digest(rows) or data.get("parent") != requirement["requirement_id"]:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CORRUPT: " + str(path))
            source = str(requirement.get("source_span", {}).get("text") or requirement["statement"])
            if data.get("source_sha256") != hashlib.sha256(source.encode("utf-8")).hexdigest():
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_SOURCE_DRIFT: " + str(path))
            result = self._verify_rows(requirement, rows)
            print("CONSTRUCTION_BEHAVIOR_CHECKPOINT: restored requirement=" +
                  requirement["requirement_id"] + " behaviors=" + str(len(result)), flush=True)
            return result

        # Transparently salvage completed requirements from the v2 shared-map
        # cache. The old key is reconstructed exactly, and is NEVER rewritten.
        if self.legacy_path.is_file():
            data = self._read_envelope(
                self.legacy_path, schema=_LEGACY_SCHEMA,
                identity=_digest({
                    "schema": _LEGACY_SCHEMA,
                    "prompt": self._prompt,
                    "requirements": self._requirements,
                    "model_id": self._model_id,
                }),
            )
            rows = data.get("rows")
            if not isinstance(rows, dict) or data.get("digest") != _digest(rows):
                raise ValueError("CONSTRUCTION_BEHAVIOR_LEGACY_CORRUPT")
            cached = rows.get(requirement["requirement_id"])
            if cached is not None:
                result = self._verify_rows(requirement, cached)
                self.save(requirement, result)
                print("CONSTRUCTION_BEHAVIOR_CHECKPOINT: migrated_v2 requirement=" +
                      requirement["requirement_id"], flush=True)
                return result
        return None

    def save(self, requirement, behaviors):
        if not self.enabled:
            return
        parent = requirement["requirement_id"]
        data = [behavior.to_dict() for behavior in behaviors]
        self._verify_rows(requirement, data)
        path = self._path(requirement)
        # Atomic file publication under claim() means a completed requirement
        # can never be lost to another requirement's snapshot write.
        if path.exists():
            present = self._read_envelope(path, schema=_SCHEMA, identity=self.identity)
            if present.get("digest") == _digest(present.get("rows")) and (
                present.get("rows") == data
            ):
                return
            raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_DIFFERENT_ACCEPTED_RECORD: " + parent)
        source = str(requirement.get("source_span", {}).get("text") or requirement["statement"])
        output = {
            "schema": _SCHEMA, "identity": self.identity, "parent": parent,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "rows": data, "digest": _digest(data),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(
                output, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n", encoding="utf-8")
            with model_execution_publication():
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
