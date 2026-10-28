"""Persisted, behavior-owned work contracts for small-model construction."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import PurePosixPath


def _relative_path(value: str) -> str:
    if value and (not isinstance(value, str) or "\\" in value or
                  PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts):
        raise ValueError("CONSTRUCTION_PATH_INVALID")
    return value


@dataclass(frozen=True)
class BehaviorRequirement:
    behavior_id: str
    requirement_id: str
    intent: str
    source_quote: str
    trigger: str
    effects: str
    state_reads: tuple[str, ...] = ()
    state_writes: tuple[str, ...] = ()
    source_status: str = "grounded"
    runtime_state_reads: tuple[str, ...] = ()
    runtime_state_writes: tuple[str, ...] = ()
    # Optional exact inspected Fabric listener. Not a model-invented callback:
    # the compilation gate resolves this against target bytecode.
    event_binding: str = ""

    def __post_init__(self):
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.behavior_id, self.requirement_id, self.intent, self.source_quote, self.effects)):
            raise ValueError("CONSTRUCTION_BEHAVIOR_IDENTITY_REQUIRED")
        if self.source_status not in {"grounded", "source_preserved_unresolved"}:
            raise ValueError("CONSTRUCTION_BEHAVIOR_SOURCE_STATUS_INVALID")

    def to_dict(self):
        # JSON checkpoint files always store arrays, not Python tuples.
        # Canonicalize before comparing accepted rows so a saved record can
        # be loaded and saved again without a false identity conflict.
        row = asdict(self)
        row["state_reads"] = list(self.state_reads)
        row["state_writes"] = list(self.state_writes)
        if not self.event_binding:
            row.pop("event_binding")
        # Keep old behavior checkpoints byte-compatible when no world/state
        # distinction was needed; new runtime obligations are explicit data.
        for field_name in ("runtime_state_reads", "runtime_state_writes"):
            if getattr(self, field_name):
                row[field_name] = list(getattr(self, field_name))
            else:
                row.pop(field_name)
        return row


@dataclass(frozen=True)
class ConstructionTask:
    task_id: str
    requirement_ids: tuple[str, ...]
    kind: str
    intent: str
    depends_on: tuple[str, ...] = ()
    source_refs: tuple[dict, ...] = ()
    target_path: str = ""
    owner_symbol: str = ""
    input_symbols: tuple[str, ...] = ()
    output_symbols: tuple[str, ...] = ()
    state_reads: tuple[str, ...] = ()
    state_writes: tuple[str, ...] = ()
    trigger: str = ""
    effects: str = ""
    allowed_writes: tuple[str, ...] = ()
    expected_old_hashes: dict | None = None
    verification_ids: tuple[str, ...] = ()
    status: str = "planned"
    behavior_id: str = ""

    def __post_init__(self):
        if not self.task_id or not self.requirement_ids or not self.intent.strip():
            raise ValueError("CONSTRUCTION_TASK_IDENTITY_REQUIRED")
        if self.kind not in {"glue", "logic", "resource", "verify"}:
            raise ValueError("CONSTRUCTION_TASK_KIND")
        if self.status not in {"planned", "ready", "running", "verified", "failed", "blocked"}:
            raise ValueError("CONSTRUCTION_TASK_STATUS")
        if self.task_id in self.depends_on:
            raise ValueError("CONSTRUCTION_TASK_SELF_DEPENDENCY")
        _relative_path(self.target_path)
        for path in self.allowed_writes:
            _relative_path(path)
        hashes = deepcopy(self.expected_old_hashes or {})
        if set(hashes) - set(self.allowed_writes):
            raise ValueError("CONSTRUCTION_REPLACEMENT_OUTSIDE_WRITE_SCOPE")
        object.__setattr__(self, "expected_old_hashes", hashes)

    def to_dict(self):
        row = asdict(self)
        for key, value in row.items():
            if isinstance(value, tuple):
                row[key] = list(value)
        return {"schema_version": "mmm/construction-task-v1", **row}

    @classmethod
    def from_dict(cls, row):
        row = deepcopy(dict(row))
        if row.pop("schema_version", None) != "mmm/construction-task-v1":
            raise ValueError("CONSTRUCTION_TASK_SCHEMA")
        if set(row) != {field.name for field in fields(cls)}:
            raise ValueError("CONSTRUCTION_TASK_FIELDS")
        for name in ("requirement_ids", "depends_on", "source_refs", "input_symbols", "output_symbols",
                     "state_reads", "state_writes", "allowed_writes", "verification_ids"):
            row[name] = tuple(row[name])
        return cls(**row)


def validate_task_graph(tasks):
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("CONSTRUCTION_TASK_DUPLICATE")
    followers = {key: [] for key in by_id}
    indegree = {}
    for task in tasks:
        if set(task.depends_on) - set(by_id):
            raise ValueError("CONSTRUCTION_TASK_DEPENDENCY_MISSING")
        indegree[task.task_id] = len(set(task.depends_on))
        for parent in set(task.depends_on):
            followers[parent].append(task.task_id)
    ready = deque(key for key, count in indegree.items() if not count)
    visited = 0
    while ready:
        parent = ready.popleft()
        visited += 1
        for child in followers[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(tasks):
        raise ValueError("CONSTRUCTION_TASK_CYCLE")
