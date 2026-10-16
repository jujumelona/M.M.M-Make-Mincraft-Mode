"""Durable, hash-bound small-model operation checkpoints.

An incomplete node author can resume without discarding accepted work. A
validated graph can be rendered again but is not gameplay verification.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .construction_state import PROTOCOL_VERSION
from .model_concurrency import model_execution_publication

_LOCK = RLock()
_MEMBER_FIELDS = (
    'kind', 'owner', 'name', 'descriptor', 'parameters', 'return_type',
    'static', 'final', 'visibility', 'access', 'annotations', 'throws',
    'class_superclass', 'class_interfaces', 'side',
)


def _sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_verified_record(path, error):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    record = envelope.get("record")
    if not isinstance(record, dict) or _sha(record) != envelope.get("sha256"):
        raise ValueError(error)
    return record


def _publish_json(path, serialized):
    """Publish a complete checkpoint only after its content is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        with model_execution_publication():
            temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def planned_declarations_fingerprint(symbols, declaration_ids):
    # A body can call another planned member. Its receiver and result contracts
    # must be stable too, even if this body's own signature did not change.
    return _sha([(key, {field: symbols[key][field] for field in _MEMBER_FIELDS
                       if field in symbols[key]}) for key in sorted(declaration_ids)])


def construction_input_fingerprint(spec, authoring_contract):
    # Identity/registry paths, paged configuration and canonical rules are body
    # inputs too. Equal signatures alone do not certify an unchanged behavior.
    from .construction_implementation import construction_implementation_fingerprint
    return _sha({'specification': spec, 'authoring_contract': authoring_contract,
                 'implementation': construction_implementation_fingerprint()})


class ConstructionNodeCheckpoint:
    def __init__(self, project_root, task, *, target, prerequisite_hashes,
                 declaration, owner_header, planned_declaration_hash, construction_input_hash):
        self.root = Path(project_root).resolve()
        self.task = task
        self.declaration = deepcopy(declaration)
        if (declaration.get('owner') != owner_header.get('owner')
                or owner_header.get('kind') != 'CLASS'):
            raise ValueError('CONSTRUCTION_CHECKPOINT_DECLARATION_INVALID')
        # Bind executable semantics, not descriptive/protocol metadata. A
        # signature with different parameter names has the same symbol ID,
        # but its accepted parameter nodes are not interchangeable.
        contract = {
            'member': {key: declaration[key] for key in _MEMBER_FIELDS if key in declaration},
            'owner_header': {key: owner_header[key] for key in (
                'kind', 'owner', 'name', 'access', 'superclass', 'interfaces', 'side',
            ) if key in owner_header},
        }
        # Unbound v1 records cannot prove their original signature. They stay
        # intact but never seed a new, declaration-bound graph. The protocol
        # version is deliberately absent so equal declarations keep their work.
        self.key = _sha({"schema": "mmm/construction-nodes-v2",
                         "task": task.to_dict(), "target": target,
                         "prerequisite_hashes": prerequisite_hashes,
                         "declaration_contract": contract,
                         "planned_declaration_hash": planned_declaration_hash,
                         "construction_input_hash": construction_input_hash})
        self.path = self.root / ".mmm" / "construction_nodes" / (self.key + ".json")
        self.author_state_path = self.path.with_suffix(
            '.author_state.' + _sha(PROTOCOL_VERSION)[:12] + '.json')

    def load(self):
        with _LOCK:
            if not self.path.exists():
                return (), ""
            record = _read_verified_record(self.path, "CONSTRUCTION_CHECKPOINT_CORRUPT")
            if record.get("key") != self.key or record.get("task_id") != self.task.task_id:
                raise ValueError("CONSTRUCTION_CHECKPOINT_BINDING_MISMATCH")
            if record.get("stage") not in {"authoring", "validated"}:
                raise ValueError("CONSTRUCTION_CHECKPOINT_STAGE_INVALID")
            if not isinstance(record.get("nodes"), list):
                raise ValueError("CONSTRUCTION_CHECKPOINT_NODES_INVALID")
            return tuple(record["nodes"]), record["stage"]

    def save(self, task, nodes, stage, *, author_state=None, expected_graph_sha256=None):
        if task.task_id != self.task.task_id or stage not in {"authoring", "validated"}:
            raise ValueError("CONSTRUCTION_CHECKPOINT_TASK_MISMATCH")
        record = {"key": self.key, "task_id": task.task_id,
                  "stage": stage, "nodes": [dict(n) for n in nodes]}
        if author_state is not None:
            if (author_state.get('protocol') != PROTOCOL_VERSION
                    or author_state.get('phase') != 'operation_graph'
                    or author_state.get('revision') != len(nodes)):
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_BINDING_MISMATCH')
            record['author_state'] = deepcopy(author_state)
        envelope = {"record": record, "sha256": _sha(record)}
        with _LOCK:
            existing, old_stage = self.load()
            if old_stage == "validated" and (
                stage != "validated" or tuple(nodes) != existing
            ):
                raise ValueError("CONSTRUCTION_CHECKPOINT_ALREADY_VALIDATED")
            if expected_graph_sha256 is not None:
                from .construction_feedback import replace_node
                if _sha(existing) != expected_graph_sha256 or old_stage != 'authoring':
                    raise ValueError('CONSTRUCTION_CHECKPOINT_REVISION_CONFLICT')
                previous = {node['node_id']: node for node in existing}
                changed = [node for node in nodes if node['node_id'] in previous
                           and node != previous[node['node_id']]]
                from .construction_shared_inputs import is_shared_binding_revision
                binding = is_shared_binding_revision(existing, nodes, self.declaration)
                if not binding and (len(changed) != 1 or tuple(nodes) != replace_node(existing, changed[0])):
                    raise ValueError('CONSTRUCTION_CHECKPOINT_REVISION_INVALID')
            elif existing and tuple(nodes[:len(existing)]) != existing:
                raise ValueError("CONSTRUCTION_CHECKPOINT_FORK")
            _publish_json(self.path, json.dumps(envelope, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")))

    def recorded_source_refs(self):
        path = self.path.with_suffix(".source_refs.json")
        with _LOCK:
            if not path.exists():
                return ()
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("key") != self.key or not isinstance(data.get("refs"), list) or (
                data.get("sha256") != _sha({"key": self.key, "refs": data.get("refs")})):
                raise ValueError("CONSTRUCTION_SOURCE_LOOKUP_CHECKPOINT_INVALID")
            return tuple(data["refs"])

    def load_author_state(self, task, nodes, *, purposes_only=False):
        if task.task_id != self.task.task_id:
            raise ValueError('CONSTRUCTION_CHECKPOINT_TASK_MISMATCH')
        with _LOCK:
            record = None
            if self.author_state_path.exists():
                record = _read_verified_record(
                    self.author_state_path, "CONSTRUCTION_AUTHOR_STATE_CORRUPT")
                if record.get('key') != self.key or record.get('protocol') != PROTOCOL_VERSION:
                    raise ValueError('CONSTRUCTION_AUTHOR_STATE_BINDING_MISMATCH')
            if record is None or record.get('nodes_sha256') != _sha(nodes):
                # Graph revisions and their initial cursor are one publication.
                # A separate lookup cursor may still describe the old graph.
                if self.path.exists():
                    body = _read_verified_record(self.path, "CONSTRUCTION_CHECKPOINT_CORRUPT")
                    if body.get('key') != self.key or body.get('task_id') != self.task.task_id:
                        raise ValueError('CONSTRUCTION_CHECKPOINT_BINDING_MISMATCH')
                    state = body.get('author_state')
                    if body.get('nodes') == list(nodes) and state is not None:
                        record = {'state': state, 'nodes_sha256': _sha(nodes)}
                if record is None:
                    return None
            if purposes_only:
                state = record.get('state')
                if not isinstance(state, dict):
                    raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
                revision = state.get('revision')
                # Sequence child purposes are checkpointed before their parent
                # node. Recover only those facts from a verified graph prefix;
                # never replay that older cursor/operation after a node commit.
                if (type(revision) is not int or not 0 <= revision <= len(nodes)
                        or record.get('nodes_sha256') != _sha(nodes[:revision])):
                    return {}
                purposes = state.get('node_purposes', {})
                if not isinstance(purposes, dict):
                    raise ValueError('CONSTRUCTION_AUTHOR_STATE_INVALID')
                live = {node['node_id'] for node in nodes}
                live.update(ref for node in nodes for ref in node['input_refs'])
                return {key: value for key, value in purposes.items() if key in live}
            # A crash between node and cursor writes may leave an older cursor.
            # Keep the accepted graph; only that cursor is stale.
            if record.get('nodes_sha256') != _sha(nodes):
                return None
            return record.get('state')

    def save_author_state(self, task, nodes, state):
        if task.task_id != self.task.task_id:
            raise ValueError('CONSTRUCTION_CHECKPOINT_TASK_MISMATCH')
        record = {'key': self.key, 'protocol': PROTOCOL_VERSION,
                  'nodes_sha256': _sha(nodes), 'state': state}
        with _LOCK:
            existing, _stage = self.load()
            if tuple(nodes) != existing:
                raise ValueError('CONSTRUCTION_AUTHOR_STATE_GRAPH_MISMATCH')
            envelope = {'record': record, 'sha256': _sha(record)}
            _publish_json(self.author_state_path, json.dumps(envelope,
                ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def record_source_read(self, task, source_ref):
        if task.task_id != self.task.task_id or not source_ref:
            raise ValueError("CONSTRUCTION_SOURCE_LOOKUP_TASK_MISMATCH")
        path = self.path.with_suffix(".source_refs.json")
        with _LOCK:
            refs = set(self.recorded_source_refs())
            refs.add(source_ref)
            body = {"key": self.key, "refs": sorted(refs)}
            record = {**body, "sha256": _sha(body)}
            _publish_json(path, json.dumps(record, sort_keys=True))
