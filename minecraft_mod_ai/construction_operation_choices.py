"""Resolve current-slot evidence before selecting its typed Java operation."""
from __future__ import annotations

import string

from .construction_author import ID, obj
from .construction_declaration_choices import action_schema, binding_lookup_schemas, symbol_lookup_schemas
from .construction_state import PAGE_SIZE
from .java_operation_ir import OPERATIONS, EXPRESSIONS, STATEMENTS
from .construction_operation_contracts import named_values, pending_input_contracts, result_fits

NODE_ID_ALPHABET = string.ascii_letters + string.digits + '_.:-'


def operation_next_schema(state, *, nodes, declaration, can_complete, context_branches=()):
    actions = list(OPERATIONS)
    if declaration['kind'] != 'CONSTRUCTOR' or any(row['op'] == 'constructor_call' for row in nodes):
        actions.remove('constructor_call')
    values = named_values(nodes, declaration, state.symbols)
    if not values['parameter']:
        actions.remove('parameter')
    if not values['local']:
        actions.remove('local')
    if declaration.get('static'):
        actions.remove('this')
    pending = pending_input_contracts(nodes, declaration, state.symbols)
    branches = []
    if state.selection_goal is None:
        raise ValueError('CONSTRUCTION_SELECTION_GOAL_REQUIRED')
    slot = state.selection_goal['node_id']
    if pending:
        # An accepted API/control node creates concrete input obligations.
        # Fill returned slots before selecting an unrelated new root.
        for key, consumers in pending.items():
            if key != slot:
                continue
            allowed = set(actions)
            for row in consumers:
                allowed &= EXPRESSIONS if row['mode'] == 'expression' else STATEMENTS
                if row.get('allowed_operations'):
                    allowed &= set(row['allowed_operations'])
            if state.evidence_index is not None:
                for action in ('parameter', 'local'):
                    if not any(result_fits(type_, consumers, state.evidence_index) for _name, type_ in values[action]):
                        allowed.discard(action)
                if not result_fits(declaration['owner'], consumers, state.evidence_index):
                    allowed.discard('this')
                for action, type_ in (('array_length', 'int'), ('instanceof', 'boolean')):
                    if not result_fits(type_, consumers, state.evidence_index):
                        allowed.discard(action)
            if allowed:
                branches.append(obj({'action': {'enum': sorted(allowed)}, 'node_id': {'const': key}, 'purpose': ID}))
    else:
        root_actions = set(actions) & (EXPRESSIONS if declaration['kind'] == 'FIELD' else STATEMENTS)
        branches.append(obj({'action': {'enum': sorted(root_actions)}, 'node_id': {'const': slot}, 'purpose': ID}))
    if can_complete:
        branches.append(action_schema('complete', {'purpose': ID}))
    # Missing information is a literal evidence action for this same slot.
    # A prose request to inspect it cannot claim a renderer deficiency.
    return {'oneOf': [*branches, *context_branches]}


def _index_schema(state, kind, query):
    branches = []
    if (kind, query, 0) not in state.requests:
        branches.append(action_schema(kind, {'query': {'const': query}, 'offset': {'const': 0}}))
    for row in state.lookup_successors(kind):
        if row['query'] == query:
            branches.append(action_schema(kind, {'query': {'const': query}, 'offset': {'const': row['next_offset']}}))
    return branches


def operation_context_branches(state, *, nodes, source_units, pinned_source):
    branches = [*binding_lookup_schemas(state), *symbol_lookup_schemas(state)]
    offset = state.next_declaration_offset()
    if offset is not None:
        branches.append(action_schema('declarations', {'query': {'const': ''}, 'offset': {'const': offset}}))
    if nodes:
        branches.extend(_index_schema(state, 'node_index', ''))
        visible = {row['node_id'] for row in nodes[-PAGE_SIZE:]}
        referenced = {ref for row in nodes for ref in row['input_refs']}
        visible.update(row['node_id'] for row in nodes if row['node_id'] not in referenced)
        for kind, query, offset in state.requests:
            if kind == 'node_index' and not query:
                visible.update(node['node_id'] for node in nodes[offset:offset + PAGE_SIZE])
        unread = sorted(key for key in visible if ('nodes', key, 0) not in state.requests)
        if unread:
            branches.append(action_schema('nodes', {'query': {'enum': unread}, 'offset': {'const': 0}}))
    if source_units and state.symbol_goal is not None:
        goal = state.evidence_index.root_goal(state.symbol_goal)
        branches.extend(_index_schema(state, 'sources', goal))
        pages = [state.evidence_index.source_page(goal, 0)]
        for kind, query, offset in state.requests:
            if kind == 'sources' and query == goal:
                pages.append(state.evidence_index.source_page(goal, offset))
        pointers = {(row['source_ref'], row['offset']) for page in pages for row in page['sources']}
        if pinned_source in source_units:
            pointers.add((pinned_source, 0))
        for ref, offset in sorted(pointers):
            if ('source', ref, offset) not in state.requests:
                branches.append(action_schema('source', {'query': {'const': ref}, 'offset': {'const': offset}}))
    return branches
