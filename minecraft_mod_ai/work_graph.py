from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from .complete_spec import CompleteProposal, ProductionModule
from .research_ledger import is_research_shard
from .scale_policy import ScalePolicy
from .spec import canonical_json


class WorkGraphError(RuntimeError):
    pass

class WorkState(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    INPUT_REQUIRED = 'input_required'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

@dataclass(frozen=True)
class WorkNode:
    node_id: str
    stage: str
    input_hash: str
    dependencies: tuple[str, ...]
    payload: dict[str, Any]
    resource_class: str = 'cpu_io'

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "stage": self.stage,
            "input_hash": self.input_hash,
            "dependencies": self.dependencies,
            "payload": self.payload,
            "resource_class": self.resource_class,
        }

@dataclass(frozen=True)
class WorkGraphPlan:
    schema_version: str
    proposal_hash: str
    graph_hash: str
    module_count: int
    nodes: tuple[WorkNode, ...]

    def to_dict(self, *, include_payloads: bool=True) -> dict[str, Any]:
        nodes = []
        for node in self.nodes:
            rendered = node.to_dict()
            if not include_payloads:
                rendered['payload'] = {'kind': node.payload.get('kind', ''), 'member_count': len(node.payload.get('members', []))}
            nodes.append(rendered)
        return {'schema_version': self.schema_version, 'proposal_hash': self.proposal_hash, 'graph_hash': self.graph_hash, 'module_count': self.module_count, 'node_count': len(self.nodes), 'nodes': nodes}

def build_production_work_plan(proposal: CompleteProposal, *, policy: ScalePolicy | None=None, modules: Sequence[ProductionModule] | None=None) -> WorkGraphPlan:
    """Compile a proposal into deterministic, dependency-aware work shards.

    The proposal has no global content-count limit. Only each work unit is bounded,
    so adding content creates more nodes instead of growing one model/tool call.
    """
    policy = policy or ScalePolicy.from_environment()
    proposal.validate(policy=policy)
    proposal_hash = proposal.approval_hash or proposal.calculate_hash()
    selected_modules = tuple(proposal.modules) if modules is None else tuple(modules)
    if modules is not None:
        for module in selected_modules:
            module.validate(policy=policy)
    selected_ids = {module.module_id for module in selected_modules}
    if len(selected_ids) != len(selected_modules):
        raise WorkGraphError('Production work modules contain duplicate IDs.')
    for module in selected_modules:
        missing = set(module.depends_on) - selected_ids
        if missing:
            raise WorkGraphError(f'Production work module {module.module_id} references missing dependencies: {sorted(missing)}')
    ordered = _topological_modules(selected_modules)
    from .platform_catalog import adapter_for_lock_values
    adapter = adapter_for_lock_values(proposal.base_proposal.spec.platform)
    deterministic_module_kinds = frozenset(
        str(kind).strip()
        for kind in getattr(adapter, 'deterministic_module_kinds', ())
        if str(kind).strip()
    )
    nodes: list[WorkNode] = [_node('prepare-project', 'prepare', (), {'kind': 'prepare', 'proposal_hash': proposal_hash, 'existing_input_sha256': proposal.existing_input_sha256})]
    module_node: dict[str, str] = {}
    exclusive_anchor_node: dict[str, str] = {}
    generated_nodes: list[str] = []
    for stage, members in _module_shards(
        ordered,
        policy=policy,
        deterministic_module_kinds=deterministic_module_kinds,
        artifact_owners=frozenset(
            job["owner_module"] for job in proposal.game_design.get("_artifact_jobs", ())
        ),
    ):
        node_id = f'generate-{stage}-{len(generated_nodes):08d}'
        member_ids = {module.module_id for module in members}
        dependencies = {'prepare-project'}
        for module in members:
            dependencies.update(module_node[dependency] for dependency in module.depends_on if dependency not in member_ids)
            dependencies.update(
                exclusive_anchor_node[anchor]
                for anchor in _exclusive_anchor_keys(module)
                if anchor in exclusive_anchor_node
            )
        dependencies.discard(node_id)
        payload = {'kind': 'module-shard', 'generation_stage': stage, 'members': [_module_payload(module) for module in members]}
        nodes.append(_node(node_id, f'generate:{stage}', sorted(dependencies), payload))
        generated_nodes.append(node_id)
        for module in members:
            module_node[module.module_id] = node_id
            for anchor in _exclusive_anchor_keys(module):
                exclusive_anchor_node[anchor] = node_id
    for index, assets in enumerate(_chunks(proposal.assets, max(1, policy.java_shard_size))):
        node_id = f'generate-assets-{index:08d}'
        nodes.append(_node(node_id, 'generate:assets', ('prepare-project',), {'kind': 'asset-shard', 'members': [asdict(asset) for asset in assets]}))
        generated_nodes.append(node_id)

    validation_dependencies = tuple(generated_nodes or ['prepare-project'])
    nodes.extend([_node('validate-source', 'validate:source', validation_dependencies, {'kind': 'validation', 'acceptance_tests': list(proposal.acceptance_tests)}), _node('build-project', 'build', ('validate-source',), {'kind': 'gradle-build'}), _node('validate-jar', 'validate:jar', ('build-project',), {'kind': 'jar-validation'}), _node('runtime-playtest', 'validate:runtime', ('validate-jar',), {'kind': 'runtime-playtest', 'external_runtime_required': proposal.external_runtime_required})])
    quality_nodes: list[str] = []
    contract = proposal.game_design.get('_production_contract')
    if proposal.schema_version == 'mmm/complete-proposal-v2' and isinstance(contract, dict):
        for dimension in contract.get('quality_dimension_catalog', []):
            dimension_id = str(dimension['dimension_id'])
            node_id = 'validate-quality-' + dimension_id.replace('_', '-')
            quality_dependency = 'validate-jar' if dimension_id in {'correctness', 'build', 'research'} else 'runtime-playtest'
            nodes.append(_node(node_id, 'validate:quality', (quality_dependency,), {'kind': 'quality-validation', 'dimension_id': dimension_id, 'evidence_route_ref': dimension['evidence_route_ref'], 'contract_sha256': contract['contract_sha256']}))
            quality_nodes.append(node_id)
    nodes.append(_node('package-release', 'package', tuple(quality_nodes or ['runtime-playtest']), {'kind': 'release'}))
    graph_body = {'schema_version': 'mmm/production-work-graph-v1', 'proposal_hash': proposal_hash, 'nodes': [node.to_dict() for node in nodes]}
    return WorkGraphPlan(schema_version='mmm/production-work-graph-v1', proposal_hash=proposal_hash, graph_hash=_hash_json(graph_body), module_count=len(selected_modules), nodes=tuple(nodes))

class DurableWorkLedger:
    """SQLite-backed task/checkpoint ledger for resumable large production.

    The database is the source of truth. Tool connections may disappear between
    calls; a later process can reopen the ledger, reclaim expired leases and keep
    working without replaying successful nodes.
    """
    schema_version = 'mmm/durable-work-ledger-v1'
    _sync_plan_temp_tables = ('affected_nodes', 'changed_nodes', 'desired_edges', 'desired_tasks')

    @classmethod
    def open_existing(cls, path: str | Path) -> DurableWorkLedger:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise FileNotFoundError(f'Work ledger not found: {resolved}')
        connection = sqlite3.connect(f'{resolved.as_uri()}?mode=ro', uri=True, timeout=30)
        try:
            values = {str(key): str(value) for key, value in connection.execute("\n                    SELECT key, value FROM metadata\n                    WHERE key IN ('proposal_hash', 'graph_hash')\n                    ")}
        finally:
            connection.close()
        proposal_hash = values.get('proposal_hash', '')
        if not proposal_hash:
            raise WorkGraphError('Work ledger has no proposal hash.')
        return cls(resolved, proposal_hash=proposal_hash, graph_hash=values.get('graph_hash', ''))

    def __init__(self, path: str | Path, *, proposal_hash: str, graph_hash: str='') -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proposal_hash = proposal_hash
        self.graph_hash = graph_hash
        self._initialize()

    @classmethod
    def _drop_sync_plan_temp_tables(cls, connection: sqlite3.Connection) -> None:
        for table_name in cls._sync_plan_temp_tables:
            connection.execute(f'DROP TABLE IF EXISTS {table_name}')

    def sync_plan(self, plan: WorkGraphPlan) -> dict[str, Any]:
        if plan.proposal_hash != self.proposal_hash:
            raise WorkGraphError('Work graph proposal does not match the ledger.')
        node_ids = {node.node_id for node in plan.nodes}
        if len(node_ids) != len(plan.nodes):
            raise WorkGraphError('Work graph contains duplicate node identifiers.')
        missing_dependencies = sorted({dependency for node in plan.nodes for dependency in node.dependencies if dependency not in node_ids})
        if missing_dependencies:
            raise WorkGraphError(f'Work graph contains unknown dependencies: {missing_dependencies[:8]}')
        now = time.time()
        connection = self._connect()
        with connection:
            self._drop_sync_plan_temp_tables(connection)
        try:
            with connection:
                connection.execute('BEGIN IMMEDIATE')
                stored_graph = self._meta(connection, 'graph_hash')
                connection.execute('\n                    CREATE TEMP TABLE desired_tasks(\n                        node_id TEXT PRIMARY KEY,\n                        stage TEXT NOT NULL,\n                        input_hash TEXT NOT NULL,\n                        payload_json TEXT NOT NULL\n                    )\n                    ')
                connection.execute('\n                    CREATE TEMP TABLE desired_edges(\n                        node_id TEXT NOT NULL,\n                        dependency_id TEXT NOT NULL,\n                        PRIMARY KEY(node_id, dependency_id)\n                    )\n                    ')
                connection.execute('\n                    CREATE TEMP TABLE changed_nodes(\n                        node_id TEXT PRIMARY KEY\n                    )\n                    ')
                connection.executemany('\n                    INSERT INTO desired_tasks(node_id, stage, input_hash, payload_json)\n                    VALUES (?, ?, ?, ?)\n                    ', ((node.node_id, node.stage, node.input_hash, canonical_json(node.payload)) for node in plan.nodes))
                connection.executemany('\n                    INSERT INTO desired_edges(node_id, dependency_id)\n                    VALUES (?, ?)\n                    ', ((node.node_id, dependency) for node in plan.nodes for dependency in node.dependencies))
                connection.execute('\n                    INSERT INTO changed_nodes(node_id)\n                    SELECT desired.node_id\n                    FROM desired_tasks AS desired\n                    JOIN tasks AS current USING(node_id)\n                    WHERE current.input_hash != desired.input_hash\n                    ')
                changed = tuple(str(row[0]) for row in connection.execute('SELECT node_id FROM changed_nodes ORDER BY node_id'))
                pruned = tuple(str(row[0]) for row in connection.execute('\n                        SELECT current.node_id\n                        FROM tasks AS current\n                        WHERE NOT EXISTS (\n                            SELECT 1 FROM desired_tasks AS desired\n                            WHERE desired.node_id = current.node_id\n                        )\n                        ORDER BY current.node_id\n                        '))
                connection.execute('DELETE FROM edges')
                connection.execute('\n                    DELETE FROM tasks\n                    WHERE NOT EXISTS (\n                        SELECT 1 FROM desired_tasks AS desired\n                        WHERE desired.node_id = tasks.node_id\n                    )\n                    ')
                connection.execute('\n                    INSERT INTO tasks(\n                        node_id, stage, input_hash, payload_json, state, updated_at\n                    )\n                    SELECT desired.node_id, desired.stage, desired.input_hash,\n                           desired.payload_json, ?, ?\n                    FROM desired_tasks AS desired\n                    WHERE NOT EXISTS (\n                        SELECT 1 FROM tasks AS current\n                        WHERE current.node_id = desired.node_id\n                    )\n                    ', (WorkState.PENDING.value, now))
                connection.execute('\n                    UPDATE tasks\n                    SET stage = (\n                            SELECT desired.stage FROM desired_tasks AS desired\n                            WHERE desired.node_id = tasks.node_id\n                        ),\n                        input_hash = (\n                            SELECT desired.input_hash FROM desired_tasks AS desired\n                            WHERE desired.node_id = tasks.node_id\n                        ),\n                        payload_json = (\n                            SELECT desired.payload_json FROM desired_tasks AS desired\n                            WHERE desired.node_id = tasks.node_id\n                        )\n                    WHERE EXISTS (\n                        SELECT 1 FROM desired_tasks AS desired\n                        WHERE desired.node_id = tasks.node_id\n                    )\n                    ')
                connection.execute('\n                    INSERT INTO edges(node_id, dependency_id)\n                    SELECT node_id, dependency_id FROM desired_edges\n                    ')
                connection.execute('\n                    CREATE TEMP TABLE affected_nodes AS\n                    WITH RECURSIVE affected(node_id) AS (\n                        SELECT node_id FROM changed_nodes\n                        UNION\n                        SELECT edges.node_id\n                        FROM edges JOIN affected\n                          ON edges.dependency_id = affected.node_id\n                    )\n                    SELECT node_id FROM affected\n                    ')
                connection.execute('\n                    UPDATE tasks\n                    SET state = ?, output_hash = NULL, receipt_json = NULL,\n                        error = NULL, lease_owner = NULL, lease_until = NULL,\n                        updated_at = ?\n                    WHERE EXISTS (\n                        SELECT 1 FROM affected_nodes\n                        WHERE affected_nodes.node_id = tasks.node_id\n                    )\n                    ', (WorkState.PENDING.value, now))
                connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('graph_hash', ?)", (plan.graph_hash,))
                connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('module_count', ?)", (str(plan.module_count),))
        finally:
            with connection:
                self._drop_sync_plan_temp_tables(connection)
        self.graph_hash = plan.graph_hash
        return {'schema_version': 'mmm/work-plan-sync-v1', 'graph_hash': plan.graph_hash, 'previous_graph_hash': stored_graph, 'node_count': len(plan.nodes), 'invalidated_nodes': tuple(sorted(changed)), 'pruned_nodes': tuple(sorted(pruned))}

    sync_plan._mmm_reusable_connection_sync_plan = True  # type: ignore[attr-defined]

    def claim_ready(self, worker_id: str, *, stages: Sequence[str]=(), lease_seconds: int=900) -> dict[str, Any] | None:
        if not worker_id.strip():
            raise WorkGraphError('worker_id must not be empty.')
        if lease_seconds < 1:
            raise WorkGraphError('lease_seconds must be positive.')
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute("\n                UPDATE tasks\n                SET state = ?, lease_owner = NULL, lease_until = NULL,\n                    error = 'expired worker lease', updated_at = ?\n                WHERE state = ? AND lease_until IS NOT NULL AND lease_until < ?\n                ", (WorkState.PENDING.value, now, WorkState.RUNNING.value, now))
            stage_sql = ''
            params: list[Any] = [WorkState.PENDING.value, WorkState.SUCCEEDED.value]
            if stages:
                placeholders = ','.join('?' for _ in stages)
                stage_sql = f' AND task.stage IN ({placeholders})'
                params.extend(stages)
            row = connection.execute(f'\n                SELECT task.node_id\n                FROM tasks AS task\n                WHERE task.state = ?\n                  AND NOT EXISTS (\n                    SELECT 1\n                    FROM edges\n                    JOIN tasks AS dependency\n                      ON dependency.node_id = edges.dependency_id\n                    WHERE edges.node_id = task.node_id\n                      AND dependency.state != ?\n                  )\n                  {stage_sql}\n                ORDER BY task.node_id\n                LIMIT 1\n                ', tuple(params)).fetchone()
            if row is None:
                connection.commit()
                return None
            node_id = str(row[0])
            connection.execute('\n                UPDATE tasks\n                SET state = ?, attempt = attempt + 1, lease_owner = ?,\n                    lease_until = ?, error = NULL, updated_at = ?\n                WHERE node_id = ? AND state = ?\n                ', (WorkState.RUNNING.value, worker_id, now + lease_seconds, now, node_id, WorkState.PENDING.value))
            connection.commit()
            return self.task(node_id)

    def task(self, node_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute('\n                SELECT node_id, stage, input_hash, payload_json, state, attempt,\n                       lease_owner, lease_until, output_hash, receipt_json, error, updated_at\n                FROM tasks WHERE node_id = ?\n                ', (node_id,)).fetchone()
            if row is None:
                raise WorkGraphError(f'Unknown work node: {node_id}')
            dependencies = tuple(row[0] for row in connection.execute('SELECT dependency_id FROM edges WHERE node_id = ? ORDER BY dependency_id', (node_id,)))
        return {'node_id': row[0], 'stage': row[1], 'input_hash': row[2], 'payload': json.loads(row[3]), 'state': row[4], 'attempt': row[5], 'lease_owner': row[6], 'lease_until': row[7], 'output_hash': row[8], 'receipt': json.loads(row[9]) if row[9] else None, 'error': row[10], 'updated_at': row[11], 'dependencies': dependencies}

    def cached_receipt(self, node_id: str, *, input_hash: str | None=None) -> dict[str, Any] | None:
        with self._connect() as connection:
            if input_hash is None:
                row = connection.execute('SELECT input_hash, state, receipt_json FROM tasks WHERE node_id = ?', (node_id,)).fetchone()
            else:
                row = connection.execute('SELECT input_hash, state, receipt_json FROM tasks WHERE node_id = ? AND input_hash = ?', (node_id, input_hash)).fetchone()
        if row is None or row[1] != WorkState.SUCCEEDED.value or not row[2]:
            return None
        return json.loads(row[2])

    def begin(self, node_id: str, *, worker_id: str='local', lease_seconds: int=900) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT state FROM tasks WHERE node_id = ?', (node_id,)).fetchone()
            if row is None:
                raise WorkGraphError(f'Unknown work node: {node_id}')
            if row[0] not in {WorkState.PENDING.value, WorkState.RUNNING.value}:
                raise WorkGraphError(f'Work node is not pending: {node_id} ({row[0]})')
            connection.execute('UPDATE tasks SET state = ?, attempt = CASE WHEN state = ? THEN attempt + 1 ELSE attempt END, lease_owner = ?, lease_until = ?, error = NULL, updated_at = ? WHERE node_id = ?', (WorkState.RUNNING.value, WorkState.PENDING.value, worker_id, now + lease_seconds, now, node_id))
            connection.commit()

    def succeed(self, node_id: str, receipt: dict[str, Any]) -> None:
        output_hash = _hash_json(receipt)
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT state FROM tasks WHERE node_id = ?', (node_id,)).fetchone()
            if row is None:
                raise WorkGraphError(f'Unknown work node: {node_id}')
            if row[0] == WorkState.SUCCEEDED.value:
                existing = connection.execute('SELECT output_hash FROM tasks WHERE node_id = ?', (node_id,)).fetchone()
                if existing is not None and existing[0] == output_hash:
                    connection.commit()
                    return
                raise WorkGraphError(f'Work node already succeeded with different output: {node_id}')
            if row[0] != WorkState.RUNNING.value:
                raise WorkGraphError(f'Work node is not running: {node_id} ({row[0]})')
            connection.execute('UPDATE tasks SET state = ?, output_hash = ?, receipt_json = ?, error = NULL, lease_owner = NULL, lease_until = NULL, updated_at = ? WHERE node_id = ?', (WorkState.SUCCEEDED.value, output_hash, canonical_json(receipt), now, node_id))
            connection.commit()

    def fail(self, node_id: str, error: str, *, input_required: bool=False) -> None:
        now = time.time()
        state = WorkState.INPUT_REQUIRED.value if input_required else WorkState.FAILED.value
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT state FROM tasks WHERE node_id = ?', (node_id,)).fetchone()
            if row is None:
                raise WorkGraphError(f'Unknown work node: {node_id}')
            if row[0] == WorkState.SUCCEEDED.value:
                raise WorkGraphError(f'Cannot fail an already succeeded work node: {node_id}')
            connection.execute('UPDATE tasks SET state = ?, error = ?, lease_owner = NULL, lease_until = NULL, updated_at = ? WHERE node_id = ?', (state, error, now, node_id))
            connection.commit()

    def retry(self, node_id: str) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT state FROM tasks WHERE node_id = ?', (node_id,)).fetchone()
            if row is None:
                raise WorkGraphError(f'Unknown work node: {node_id}')
            if row[0] not in {WorkState.FAILED.value, WorkState.INPUT_REQUIRED.value, WorkState.CANCELLED.value}:
                raise WorkGraphError(f'Work node is not retryable: {node_id} ({row[0]})')
            connection.execute('UPDATE tasks SET state = ?, error = NULL, lease_owner = NULL, lease_until = NULL, updated_at = ? WHERE node_id = ?', (WorkState.PENDING.value, now, node_id))
            connection.commit()

    def invalidate(self, node_id: str) -> tuple[str, ...]:
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            affected = self._invalidate_many(connection, (node_id,))
            connection.commit()
        return affected

    def request_cancel(self) -> None:
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('cancel_requested', ?)", (str(time.time()),))
            connection.commit()

    def raise_if_cancelled(self) -> None:
        with self._connect() as connection:
            if self._meta(connection, 'cancel_requested'):
                raise WorkGraphError('Run cancellation requested.')

    def begin_checkpoint(self, checkpoint_id: str, *, stage: str, input_hash: str) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT input_hash, state FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,)).fetchone()
            if row is None:
                connection.execute('INSERT INTO checkpoints(checkpoint_id, stage, input_hash, state, attempt, updated_at) VALUES (?, ?, ?, ?, ?, ?)', (checkpoint_id, stage, input_hash, WorkState.RUNNING.value, 1, now))
            else:
                connection.execute('UPDATE checkpoints SET stage = ?, input_hash = ?, state = ?, attempt = attempt + 1, output_hash = NULL, receipt_json = NULL, error = NULL, updated_at = ? WHERE checkpoint_id = ?', (stage, input_hash, WorkState.RUNNING.value, now, checkpoint_id))
            connection.commit()

    def cached_checkpoint(self, checkpoint_id: str, *, input_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute('SELECT input_hash, state, receipt_json FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,)).fetchone()
        if row is None or row[0] != input_hash or row[1] != WorkState.SUCCEEDED.value or not row[2]:
            return None
        return json.loads(row[2])

    def succeed_checkpoint(self, checkpoint_id: str, *, input_hash: str, receipt: dict[str, Any]) -> None:
        output_hash = _hash_json(receipt)
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT input_hash, state FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,)).fetchone()
            if row is None or row[0] != input_hash or row[1] != WorkState.RUNNING.value:
                raise WorkGraphError(f'Checkpoint is not running with the expected input: {checkpoint_id}')
            connection.execute('UPDATE checkpoints SET state = ?, output_hash = ?, receipt_json = ?, error = NULL, updated_at = ? WHERE checkpoint_id = ?', (WorkState.SUCCEEDED.value, output_hash, canonical_json(receipt), now, checkpoint_id))
            connection.commit()

    def fail_checkpoint(self, checkpoint_id: str, *, input_hash: str, error: str) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT input_hash FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,)).fetchone()
            if row is not None and row[0] == input_hash:
                connection.execute('UPDATE checkpoints SET state = ?, error = ?, updated_at = ? WHERE checkpoint_id = ?', (WorkState.FAILED.value, error, now, checkpoint_id))
            connection.commit()

    def page(self, *, cursor: str='', limit: int=100, state: WorkState | str | None=None) -> dict[str, Any]:
        if limit < 1:
            raise WorkGraphError('limit must be positive.')
        normalized_state = WorkState(state).value if state is not None else None
        params: list[Any] = [cursor]
        state_sql = ''
        if normalized_state is not None:
            state_sql = ' AND task.state = ?'
            params.append(normalized_state)
        params.append(limit + 1)
        with self._connect() as connection:
            rows = connection.execute(f'\n            SELECT task.node_id, task.stage, task.input_hash, task.payload_json,\n                   task.state, task.attempt, task.lease_owner, task.lease_until,\n                   task.output_hash, task.receipt_json, task.error, task.updated_at\n            FROM tasks AS task\n            WHERE task.node_id > ? {state_sql}\n            ORDER BY task.node_id\n            LIMIT ?\n            ', tuple(params)).fetchall()
            page_rows = rows[:limit]
            dependencies = {str(row[0]): tuple(dep[0] for dep in connection.execute('SELECT dependency_id FROM edges WHERE node_id = ? ORDER BY dependency_id', (row[0],))) for row in page_rows}
        page = [
            {
                'node_id': row[0],
                'stage': row[1],
                'input_hash': row[2],
                'payload': json.loads(row[3]),
                'state': row[4],
                'attempt': row[5],
                'lease_owner': row[6],
                'lease_until': row[7],
                'output_hash': row[8],
                'receipt': json.loads(row[9]) if row[9] else None,
                'error': row[10],
                'updated_at': row[11],
                'dependencies': dependencies[str(row[0])],
            }
            for row in page_rows
        ]
        return {'schema_version': 'mmm/work-task-page-v1', 'tasks': page, 'next_cursor': page[-1]['node_id'] if len(rows) > limit else ''}

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {row[0]: row[1] for row in connection.execute('SELECT state, COUNT(*) FROM tasks GROUP BY state')}
            total = sum(counts.values())
            checkpoint_counts = {row[0]: row[1] for row in connection.execute('SELECT state, COUNT(*) FROM checkpoints GROUP BY state')}
            cancel_requested = self._meta(connection, 'cancel_requested')
            graph_hash = self._meta(connection, 'graph_hash')
            module_count = self._meta(connection, 'module_count')
        completed = counts.get(WorkState.SUCCEEDED.value, 0)
        return {'schema_version': 'mmm/work-ledger-summary-v1', 'proposal_hash': self.proposal_hash, 'graph_hash': graph_hash, 'module_count': int(module_count or '0'), 'task_count': total, 'counts': counts, 'checkpoint_counts': checkpoint_counts, 'cancel_requested': cancel_requested or None, 'progress': 1.0 if total == 0 else round(completed / total, 6)}

    def export_receipts(self, path: str | Path) -> Path:
        """Stream a portable audit log without loading every receipt at once."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(canonical_json({'record_type': 'summary', 'value': self.summary()}) + '\n')
            with self._connect() as connection:
                for row in connection.execute('\n                    SELECT node_id, stage, input_hash, state, attempt,\n                           output_hash, receipt_json, error, updated_at\n                    FROM tasks ORDER BY node_id\n                    '):
                    stream.write(canonical_json({'record_type': 'task', 'node_id': row[0], 'stage': row[1], 'input_hash': row[2], 'state': row[3], 'attempt': row[4], 'output_hash': row[5], 'receipt': json.loads(row[6]) if row[6] else None, 'error': row[7], 'updated_at': row[8]}) + '\n')
                for row in connection.execute('\n                    SELECT checkpoint_id, stage, input_hash, state, attempt,\n                           output_hash, receipt_json, error, updated_at\n                    FROM checkpoints ORDER BY checkpoint_id\n                    '):
                    stream.write(canonical_json({'record_type': 'checkpoint', 'checkpoint_id': row[0], 'stage': row[1], 'input_hash': row[2], 'state': row[3], 'attempt': row[4], 'output_hash': row[5], 'receipt': json.loads(row[6]) if row[6] else None, 'error': row[7], 'updated_at': row[8]}) + '\n')
        temporary.replace(target)
        return target

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript('\n                CREATE TABLE IF NOT EXISTS metadata(\n                    key TEXT PRIMARY KEY,\n                    value TEXT NOT NULL\n                );\n                CREATE TABLE IF NOT EXISTS tasks(\n                    node_id TEXT PRIMARY KEY,\n                    stage TEXT NOT NULL,\n                    input_hash TEXT NOT NULL,\n                    payload_json TEXT NOT NULL,\n                    state TEXT NOT NULL,\n                    attempt INTEGER NOT NULL DEFAULT 0,\n                    lease_owner TEXT,\n                    lease_until REAL,\n                    output_hash TEXT,\n                    receipt_json TEXT,\n                    error TEXT,\n                    updated_at REAL NOT NULL\n                );\n                CREATE TABLE IF NOT EXISTS edges(\n                    node_id TEXT NOT NULL REFERENCES tasks(node_id)\n                        ON DELETE CASCADE,\n                    dependency_id TEXT NOT NULL REFERENCES tasks(node_id)\n                        ON DELETE CASCADE,\n                    PRIMARY KEY(node_id, dependency_id)\n                );\n                CREATE INDEX IF NOT EXISTS tasks_state_stage\n                    ON tasks(state, stage, node_id);\n                CREATE INDEX IF NOT EXISTS edges_dependency\n                    ON edges(dependency_id, node_id);\n                CREATE TABLE IF NOT EXISTS checkpoints(\n                    checkpoint_id TEXT PRIMARY KEY,\n                    stage TEXT NOT NULL,\n                    input_hash TEXT NOT NULL,\n                    state TEXT NOT NULL,\n                    attempt INTEGER NOT NULL DEFAULT 0,\n                    output_hash TEXT,\n                    receipt_json TEXT,\n                    error TEXT,\n                    updated_at REAL NOT NULL\n                );\n                CREATE INDEX IF NOT EXISTS checkpoints_state_stage\n                    ON checkpoints(state, stage, checkpoint_id);\n                ')
            stored_schema = self._meta(connection, 'schema_version')
            stored_proposal = self._meta(connection, 'proposal_hash')
            if stored_schema and stored_schema != self.schema_version:
                raise WorkGraphError('Unsupported durable work ledger schema.')
            if stored_proposal and stored_proposal != self.proposal_hash:
                raise WorkGraphError('Existing run belongs to a different approved proposal.')
            connection.execute('INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)', ('schema_version', self.schema_version))
            connection.execute('INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)', ('proposal_hash', self.proposal_hash))
            if self.graph_hash:
                connection.execute('INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)', ('graph_hash', self.graph_hash))
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        local = getattr(self, '_mmm_sqlite_local', None)
        if local is None:
            local = threading.local()
            self._mmm_sqlite_local = local

        connection = getattr(local, 'connection', None)
        pid = getattr(local, 'pid', None)
        if connection is not None and pid != os.getpid():
            try:
                connection.close()
            except Exception:
                pass
            connection = None

        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.execute('PRAGMA foreign_keys = ON')
            connection.execute('PRAGMA journal_mode = WAL')
            connection.execute('PRAGMA synchronous = NORMAL')
            local.connection = connection
            local.pid = os.getpid()
        return connection

    _connect._mmm_thread_local_connection = True  # type: ignore[attr-defined]

    @staticmethod
    def _meta(connection: sqlite3.Connection, key: str) -> str:
        row = connection.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
        return str(row[0]) if row is not None else ''

    def _read_meta(self, key: str) -> str:
        with self._connect() as connection:
            return self._meta(connection, key)

    @staticmethod
    def _invalidate_many(connection: sqlite3.Connection, node_ids: Sequence[str]) -> tuple[str, ...]:
        if not node_ids:
            return ()
        placeholders = ','.join('?' for _ in node_ids)
        rows = connection.execute(f'\n            WITH RECURSIVE affected(node_id) AS (\n                SELECT node_id FROM tasks WHERE node_id IN ({placeholders})\n                UNION\n                SELECT edges.node_id\n                FROM edges JOIN affected\n                  ON edges.dependency_id = affected.node_id\n            )\n            SELECT node_id FROM affected ORDER BY node_id\n            ', tuple(node_ids)).fetchall()
        affected = tuple(str(row[0]) for row in rows)
        if not affected:
            return ()
        update_placeholders = ','.join('?' for _ in affected)
        connection.execute(f'\n            UPDATE tasks\n            SET state = ?, output_hash = NULL, receipt_json = NULL,\n                error = NULL, lease_owner = NULL, lease_until = NULL,\n                updated_at = ?\n            WHERE node_id IN ({update_placeholders})\n            ', (WorkState.PENDING.value, time.time(), *affected))
        return affected
T = TypeVar('T')

def run_checkpoint(ledger: DurableWorkLedger, node_id: str, action: Callable[[], T], *, encode: Callable[[T], dict[str, Any]], decode: Callable[[dict[str, Any]], T], worker_id: str='local') -> T:
    ledger.raise_if_cancelled()
    cached = ledger.cached_receipt(node_id)
    if cached is not None:
        return decode(cached)
    ledger.begin(node_id, worker_id=worker_id)
    try:
        value = action()
        ledger.succeed(node_id, encode(value))
        return value
    except Exception as exc:
        ledger.fail(node_id, f'{type(exc).__name__}: {exc}')
        raise

def run_named_checkpoint(ledger: DurableWorkLedger, checkpoint_id: str, *, stage: str, input_value: Any, action: Callable[[], T], encode: Callable[[T], dict[str, Any]], decode: Callable[[dict[str, Any]], T], validate_cached: Callable[[T], bool] | None=None) -> T:
    ledger.raise_if_cancelled()
    input_hash = _hash_json(input_value)
    cached = ledger.cached_checkpoint(checkpoint_id, input_hash=input_hash)
    if cached is not None:
        decoded = decode(cached)
        if validate_cached is None or validate_cached(decoded):
            return decoded
    ledger.begin_checkpoint(checkpoint_id, stage=stage, input_hash=input_hash)
    try:
        value = action()
        ledger.succeed_checkpoint(checkpoint_id, input_hash=input_hash, receipt=encode(value))
        return value
    except Exception as exc:
        ledger.fail_checkpoint(checkpoint_id, input_hash=input_hash, error=f'{type(exc).__name__}: {exc}')
        raise

def _content_node_is_cpu_safe(payload: dict[str, Any]) -> bool:
    members = payload.get('members')
    if not isinstance(members, list):
        return True
    for member in members:
        if not isinstance(member, dict):
            return False
        if str(member.get('kind', '')) != 'integration':
            continue
        config = member.get('config')
        if not isinstance(config, dict):
            return False
        if str(config.get('integration_type', '')) != 'mmm_local_ai_sidecar':
            return False
    return True

def _node(node_id: str, stage: str, dependencies: Iterable[str], payload: dict[str, Any]) -> WorkNode:
    kind = str(payload.get('kind', ''))
    gen_stage = str(payload.get('generation_stage', ''))
    if 'resource_class' in payload:
        res_class = payload['resource_class']
    elif kind == 'asset-shard':
        res_class = 'image_gpu'
    elif kind == 'module-shard' and gen_stage in {'content', 'system', 'entity'}:
        res_class = 'llm' if gen_stage == 'content' and not _content_node_is_cpu_safe(payload) else 'cpu_io'
    elif kind == 'module-shard' and gen_stage == 'custom':
        res_class = 'llm'
    elif stage.startswith('validate:'):
        res_class = 'commit'
    else:
        res_class = 'cpu_io'
    payload_copy = dict(payload)
    payload_copy['resource_class'] = res_class
    normalized_dependencies = tuple(sorted(set(dependencies)))
    body = {'node_id': node_id, 'stage': stage, 'dependencies': normalized_dependencies, 'payload': payload_copy}
    return WorkNode(node_id=node_id, stage=stage, input_hash=_hash_json(body), dependencies=normalized_dependencies, payload=payload_copy, resource_class=res_class)

_node._mmm_stage_parallel_generation_lanes = True  # type: ignore[attr-defined]
_node._mmm_shared_write_commit_lane = True  # type: ignore[attr-defined]

def _module_payload(module: ProductionModule) -> dict[str, Any]:
    return {'module_id': module.module_id, 'kind': module.kind, 'config': module.config, 'depends_on': list(module.depends_on), 'required_gates': list(module.required_gates)}


def _exclusive_anchor_keys(module: ProductionModule) -> tuple[str, ...]:
    """Return exact host-reserved anchors that may not execute concurrently."""

    config = module.config if isinstance(module.config, dict) else {}
    raw = config.get('owned_anchors')
    if not isinstance(raw, list):
        evidence_task = config.get('evidence_task')
        raw = evidence_task.get('owned_anchors') if isinstance(evidence_task, dict) else None
    if not isinstance(raw, list):
        return ()
    keys: list[str] = []
    for anchor in raw:
        if not isinstance(anchor, dict) or anchor.get('ownership') != 'exclusive':
            continue
        locator = str(anchor.get('locator') or '').strip()
        if locator:
            keys.append(locator)
    return tuple(dict.fromkeys(keys))

def _module_stage(
    module: ProductionModule,
    *,
    deterministic_module_kinds: frozenset[str] | None = None,
) -> str:
    if is_research_shard(module) or module.kind == 'research_shard':
        return 'content'
    if module.kind == 'integration':
        if str(module.config.get('integration_type', '')) == 'mmm_local_ai_sidecar':
            return 'content'
        return 'custom'
    if module.kind == 'custom_java' or module.config.get('implementation') == 'custom':
        return 'custom'
    if module.kind in {'entity', 'boss', 'npc'}:
        return 'entity'
    if module.kind in {'quest', 'class', 'skill', 'economy', 'shop', 'gui', 'networking', 'party', 'guild'}:
        return 'system'
    extended_kinds = {'item', 'block', 'effect', 'enchantment', 'command', 'recipe', 'tag', 'advancement', 'loot', 'tool', 'weapon', 'armor', 'food', 'crop', 'machine'}
    if module.kind in extended_kinds:
        if deterministic_module_kinds is None or module.kind in deterministic_module_kinds:
            return 'content'
        return 'custom'
    return 'custom'

def _active_llm_slots() -> int:
    raw = os.environ.get('MMM_LLAMA_ACTIVE_PARALLEL', '1').strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1

def _pipeline_shard_size(name: str, default: int, upper: int) -> int:
    raw = os.environ.get(name, '').strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(max(1, upper), value))

def _module_shards(
    modules: Sequence[ProductionModule],
    *,
    policy: ScalePolicy,
    deterministic_module_kinds: frozenset[str] | None = None,
    artifact_owners: frozenset[str] = frozenset(),
) -> Iterator[tuple[str, tuple[ProductionModule, ...]]]:
    """Emit fine-grained dependency-ready units so the outer DAG owns parallelism."""
    staged = [
        (
            module,
            "content" if module.module_id in artifact_owners else _module_stage(
                module,
                deterministic_module_kinds=deterministic_module_kinds,
            ),
        )
        for module in modules
    ]
    stage_counts: dict[str, int] = {}
    for _module, stage in staged:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    groups: list[dict[str, Any]] = []
    module_group: dict[str, int] = {}
    open_by_key: dict[tuple[str, frozenset[int]], int] = {}

    def shard_size_for(stage: str) -> int:
        # Generation stages that contain per-member loops default to one member per
        # WorkNode. This exposes dependency and validation pipelining to the outer DAG,
        # while collision anchors still serialize nodes that share mutation domains.
        if stage == 'entity':
            return _pipeline_shard_size(
                'MMM_ENTITY_PIPELINE_SHARD_SIZE',
                1,
                max(1, int(policy.entity_shard_size)),
            )
        if stage == 'content':
            return _pipeline_shard_size(
                'MMM_CONTENT_PIPELINE_SHARD_SIZE',
                1,
                max(1, int(policy.java_shard_size)),
            )
        if stage == 'system':
            return _pipeline_shard_size(
                'MMM_SYSTEM_PIPELINE_SHARD_SIZE',
                1,
                max(1, int(policy.java_shard_size)),
            )
        if stage == 'custom':
            count = max(1, stage_counts.get(stage, 1))
            slots = min(_active_llm_slots(), count)
            per_slot = (count + slots - 1) // slots
            return _pipeline_shard_size(
                'MMM_CUSTOM_PIPELINE_SHARD_SIZE',
                max(1, per_slot),
                max(1, int(policy.java_shard_size)),
            )
        return max(1, int(policy.java_shard_size))

    for module, stage in staged:
        missing = [dependency for dependency in module.depends_on if dependency not in module_group]
        if missing:
            raise WorkGraphError(
                'Module sharding requires topological order; unresolved dependencies for '
                f'{module.module_id}: {missing[:4]}'
            )

        shard_size = shard_size_for(stage)
        dependency_groups = {module_group[dependency] for dependency in module.depends_on}
        candidates: set[int] = set()

        exact_key = (stage, frozenset(dependency_groups))
        exact = open_by_key.get(exact_key)
        if exact is not None and len(groups[exact]['members']) < shard_size:
            candidates.add(exact)

        for index in dependency_groups:
            group = groups[index]
            if group['stage'] != stage or len(group['members']) >= shard_size:
                continue
            if (dependency_groups - {index}).issubset(group['external_groups']):
                candidates.add(index)

        chosen = max(candidates) if candidates else None
        if chosen is None:
            chosen = len(groups)
            external_groups = set(dependency_groups)
            groups.append(
                {
                    'stage': stage,
                    'members': [],
                    'external_groups': external_groups,
                    'first_order': len(module_group),
                }
            )
            open_by_key[(stage, frozenset(external_groups))] = chosen

        group = groups[chosen]
        group['members'].append(module)
        module_group[module.module_id] = chosen

        group_key = (str(group['stage']), frozenset(group['external_groups']))
        if len(group['members']) >= shard_size:
            if open_by_key.get(group_key) == chosen:
                open_by_key.pop(group_key, None)
        else:
            previous = open_by_key.get(group_key)
            if previous is None or chosen > previous:
                open_by_key[group_key] = chosen

    dependents: dict[int, list[int]] = {index: [] for index in range(len(groups))}
    indegree = [0] * len(groups)
    for index, group in enumerate(groups):
        dependencies = sorted(set(group['external_groups']))
        indegree[index] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(index)

    ready = sorted(
        (index for index, degree in enumerate(indegree) if degree == 0),
        key=lambda index: int(groups[index]['first_order']),
    )
    emitted = 0
    while ready:
        next_ready: set[int] = set()
        for index in ready:
            group = groups[index]
            yield str(group['stage']), tuple(group['members'])
            emitted += 1
            for dependent in dependents[index]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.add(dependent)
        ready = sorted(next_ready, key=lambda index: int(groups[index]['first_order']))

    if emitted != len(groups):
        raise WorkGraphError('Module shard dependency graph contains a cycle.')

_module_shards._mmm_dependency_wave_shards = True  # type: ignore[attr-defined]
_module_shards._mmm_entity_pipeline_granularity = True  # type: ignore[attr-defined]
_module_shards._mmm_module_level_generation = True  # type: ignore[attr-defined]

def _topological_modules(modules: Sequence[ProductionModule]) -> tuple[ProductionModule, ...]:
    import heapq
    lookup = {module.module_id: module for module in modules}
    indegree = {module.module_id: len(module.depends_on) for module in modules}
    outgoing: dict[str, list[str]] = {module.module_id: [] for module in modules}
    for module in modules:
        for dependency in module.depends_on:
            outgoing[dependency].append(module.module_id)
    ready = [node_id for node_id, value in indegree.items() if value == 0]
    heapq.heapify(ready)
    ordered: list[ProductionModule] = []
    while ready:
        node_id = heapq.heappop(ready)
        ordered.append(lookup[node_id])
        for dependent in outgoing[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(modules):
        raise WorkGraphError('Production module graph contains a cycle.')
    return tuple(ordered)

def _chunks(values: Sequence[T], size: int) -> Iterator[tuple[T, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index:index + size])

def _hash_json(value: Any) -> str:
    return 'sha256:' + hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()
