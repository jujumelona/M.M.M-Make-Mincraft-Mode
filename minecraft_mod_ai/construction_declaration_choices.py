"""Native declaration choices derived from the inspected page and owner plan.

An instruction is not a type/ownership constraint. Keep action-specific
schemas closed and expose only choices the current phase can actually accept.
"""
from __future__ import annotations

from .construction_author import ID, obj


def action_schema(action, properties=None):
    return obj({'action': {'const': action}, **(properties or {})})


def choice_actions(schema):
    return [branch['properties']['action']['const'] for branch in schema['oneOf']]


def accessible_type(state, row):
    if row.get('kind') != 'CLASS': return False
    if row.get('side') == 'CLIENT' and not state.target_path.startswith('src/client/'):
        return False
    access = row.get('access', 0)
    same_package = row.get('owner', '').rpartition('.')[0] == state.owner.rpartition('.')[0]
    return bool(access & 1 or (same_package and not access & 2))


def header_candidates(state):
    candidates = {'superclass': [], 'interface': []}
    header = state.symbols[state.owner]
    for match in state.lookup.get('matches', ()):
        key = match['symbol_id']
        row = state.symbols.get(key, {})
        if (key == state.owner or row.get('provenance') == 'construction_plan'
                or not accessible_type(state, row)):
            continue
        access = row.get('access', 0)
        if access & 0x0200:
            if key not in header['interfaces']: candidates['interface'].append(key)
        elif not access & 0x0010 and not header['superclass']:
            candidates['superclass'].append(key)
    return {kind: sorted(set(keys)) for kind, keys in candidates.items()}


def header_schema(candidates):
    branches = [action_schema('symbols', {'query': ID, 'offset': {'type': 'integer', 'minimum': 0}}),
                action_schema('complete'), action_schema('extension', {'reason': ID})]
    for kind, keys in candidates.items():
        if keys:
            branches.append(action_schema(kind, {'type_ref': {'type': 'string', 'enum': keys}}))
    return {'oneOf': branches}
