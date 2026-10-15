"""Host-owned operation observations and exact single-node graph revisions."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy


def graph_digest(nodes):
    return hashlib.sha256(json.dumps(nodes, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def replace_node(nodes, replacement):
    """Keep original roots/shared inputs; remove only newly orphaned descendants."""
    by_id = {node['node_id']: node for node in nodes}
    key = replacement['node_id']
    if key not in by_id or by_id[key] == replacement:
        raise ValueError('CONSTRUCTION_REVISION_TARGET_INVALID')
    consumed = {ref for node in nodes for ref in node['input_refs']}
    roots = [node_id for node_id in by_id if node_id not in consumed]
    by_id[key] = replacement
    live, pending = set(), list(roots)
    while pending:
        current = pending.pop()
        if current in live or current not in by_id:
            continue
        live.add(current)
        pending.extend(by_id[current]['input_refs'])
    if key not in live:
        raise ValueError('CONSTRUCTION_REVISION_TARGET_UNREACHABLE')
    return tuple(deepcopy(by_id[node['node_id']]) for node in nodes if node['node_id'] in live)


def rejection(error, nodes, *, candidate=None, phase='partial'):
    diagnostic = str(error)
    if not diagnostic.startswith('CONSTRUCTION_'):
        raise error
    return {'status': 'rejected', 'phase': phase, 'graph_sha256': graph_digest(nodes),
            'node_id': getattr(error, 'node_id', None) or (candidate or {}).get('node_id'),
            'diagnostic': diagnostic,
            'rejected_node': deepcopy(candidate),
            'validated_node_count': 0, 'deferred_node_count': 0}


def admission(nodes, graph, node_id):
    return {'status': 'admitted', 'phase': 'partial', 'graph_sha256': graph_digest(nodes),
            'node_id': node_id, 'diagnostic': '', 'rejected_node': None,
            'validated_node_count': len(graph.validated_node_ids),
            'deferred_node_count': len(graph.deferred_node_ids)}


def validate_observation(record):
    if record is None:
        return
    fields = {'status', 'phase', 'graph_sha256', 'node_id', 'diagnostic', 'rejected_node',
              'validated_node_count', 'deferred_node_count'}
    if (not isinstance(record, dict) or set(record) != fields
            or record['status'] not in {'rejected', 'admitted'}
            or record['phase'] not in {'partial', 'complete'}
            or not isinstance(record['graph_sha256'], str)
            or not re.fullmatch(r'[a-f0-9]{64}', record['graph_sha256'])
            or (record['node_id'] is not None and (not isinstance(record['node_id'], str)
                or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}', record['node_id'])))
            or not isinstance(record['diagnostic'], str)
            or (record['rejected_node'] is not None and not isinstance(record['rejected_node'], dict))
            or any(type(record[key]) is not int or record[key] < 0
                   for key in ('validated_node_count', 'deferred_node_count'))):
        raise ValueError('CONSTRUCTION_OBSERVATION_INVALID')


def observation_view(record):
    """One local result, not a growing conversation or a repeated full graph."""
    if record is None:
        return None
    result = deepcopy(record)
    node = result.get('rejected_node')
    if node:
        refs = node.get('input_refs', [])
        node['input_ref_count'] = len(refs)
        node['input_refs'] = refs[:8]
        attributes = node.get('attributes', {})
        node['attribute_sequence_counts'] = {
            key: len(value) for key, value in attributes.items() if isinstance(value, list)}
        node['attributes'] = {key: value[:8] if isinstance(value, list) else value
                              for key, value in attributes.items()}
    result['meaning'] = ('A partial type/access check is not behavior or runtime verification. '
                         'Correct the diagnostic in the same slot; keep unrelated work.')
    return result
