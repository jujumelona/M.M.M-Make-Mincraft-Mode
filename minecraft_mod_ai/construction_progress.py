"""Durable, hash-bound small-model operation checkpoints.

An incomplete node author can resume without discarding accepted work. A
validated graph can be rendered again but is not gameplay verification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4


_LOCK = RLock()


def _sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ConstructionNodeCheckpoint:
    def __init__(self, project_root, task, *, target, prerequisite_hashes):
        self.root = Path(project_root).resolve()
        self.task = task
        self.key = _sha({"schema": "mmm/construction-nodes-v1",
                         "task": task.to_dict(), "target": target,
                         "prerequisite_hashes": prerequisite_hashes})
        self.path = self.root / ".mmm" / "construction_nodes" / (self.key + ".json")

    def load(self):
        with _LOCK:
            if not self.path.exists():
                return (), ""
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
            record = envelope.get("record")
            if not isinstance(record, dict) or _sha(record) != envelope.get("sha256"):
                raise ValueError("CONSTRUCTION_CHECKPOINT_CORRUPT")
            if record.get("key") != self.key or record.get("task_id") != self.task.task_id:
                raise ValueError("CONSTRUCTION_CHECKPOINT_BINDING_MISMATCH")
            if record.get("stage") not in {"authoring", "validated"}:
                raise ValueError("CONSTRUCTION_CHECKPOINT_STAGE_INVALID")
            if not isinstance(record.get("nodes"), list):
                raise ValueError("CONSTRUCTION_CHECKPOINT_NODES_INVALID")
            return tuple(record["nodes"]), record["stage"]

    def save(self, task, nodes, stage):
        if task.task_id != self.task.task_id or stage not in {"authoring", "validated"}:
            raise ValueError("CONSTRUCTION_CHECKPOINT_TASK_MISMATCH")
        record = {"key": self.key, "task_id": task.task_id,
                  "stage": stage, "nodes": [dict(n) for n in nodes]}
        envelope = {"record": record, "sha256": _sha(record)}
        with _LOCK:
            existing, old_stage = self.load()
            if old_stage == "validated" and (
                stage != "validated" or tuple(nodes) != existing
            ):
                raise ValueError("CONSTRUCTION_CHECKPOINT_ALREADY_VALIDATED")
            if existing and tuple(nodes[:len(existing)]) != existing:
                raise ValueError("CONSTRUCTION_CHECKPOINT_FORK")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name("." + self.path.name + "." + uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps(envelope, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")), encoding="utf-8")
                temporary.replace(self.path)
            finally:
                if temporary.exists():
                    temporary.unlink()
