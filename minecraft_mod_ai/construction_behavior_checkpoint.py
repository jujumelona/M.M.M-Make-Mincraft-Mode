"""Crash-safe accepted behavior records, bound to original request and model.

Reuse only completed, source-validated requirement pages. This prevents a
late source-decomposition interruption from discarding earlier model work.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

_LOCK = RLock()
_SCHEMA = "mmm/behavior-plan-checkpoint-v2"


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
        self.identity = _digest({
            "schema": _SCHEMA, "prompt": scope.get("requested_prompt"),
            "requirements": scope.get("requirements"),
            "model_id": model_id,
        })
        self.path = root / (self.identity + ".json")
        self._rows = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("schema") != _SCHEMA or saved.get("identity") != self.identity:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_IDENTITY")
            rows = saved.get("rows")
            if not isinstance(rows, dict) or saved.get("digest") != _digest(rows):
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CORRUPT")
            self._rows = rows

    def load(self, requirement):
        if not self.enabled:
            return None
        parent = requirement["requirement_id"]
        cached = self._rows.get(parent)
        if cached is None:
            return None
        from .construction_task import BehaviorRequirement
        from .request_scope import _source_quote_bounds
        source = str(requirement.get("source_span", {}).get("text") or
                     requirement["statement"])
        output = []
        if not cached:
            raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_EMPTY")
        for row in cached:
            record = BehaviorRequirement(**{
                **row,
                "state_reads": tuple(row.get("state_reads", ())),
                "state_writes": tuple(row.get("state_writes", ())),
            })
            if record.requirement_id != parent:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_PARENT")
            if _source_quote_bounds(source, record.source_quote) is None:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_SOURCE_DRIFT")
            output.append(record)
        print("CONSTRUCTION_BEHAVIOR_CHECKPOINT: restored requirement=" +
              parent + " behaviors=" + str(len(output)), flush=True)
        return tuple(output)

    def save(self, requirement, behaviors):
        if not self.enabled:
            return
        parent = requirement["requirement_id"]
        data = [behavior.to_dict() for behavior in behaviors]
        if not data or any(row["requirement_id"] != parent for row in data):
            raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_COVERAGE")
        with _LOCK:
            if parent in self._rows:
                if self._rows[parent] != data:
                    raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CONFLICT")
                return
            if self.path.exists():
                envelope = json.loads(self.path.read_text(encoding="utf-8"))
                if envelope.get("identity") != self.identity or (
                    envelope.get("digest") != _digest(envelope.get("rows"))):
                    raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CONFLICT")
                disk_rows = envelope["rows"]
                for key, value in self._rows.items():
                    if disk_rows.get(key) != value:
                        raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CONFLICT")
                self._rows = disk_rows
            if parent in self._rows and self._rows[parent] != data:
                raise ValueError("CONSTRUCTION_BEHAVIOR_CHECKPOINT_CONFLICT")
            self._rows[parent] = data
            envelope = {"schema": _SCHEMA, "identity": self.identity,
                        "rows": self._rows, "digest": _digest(self._rows)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name("." + self.path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps(envelope, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                temporary.replace(self.path)
            finally:
                if temporary.exists():
                    temporary.unlink()
