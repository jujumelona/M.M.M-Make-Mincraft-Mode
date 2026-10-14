"""Operation decisions with selectors derived from real, unread host pages."""
from __future__ import annotations

import string

from .construction_author import ID, obj
from .construction_declaration_choices import (
    action_schema, binding_lookup_schemas, string_excluding, symbol_lookup_schemas,
)
from .construction_state import PAGE_SIZE, symbol_query_key
from .java_operation_ir import OPERATIONS

_SELECTOR_ALPHABET = ''.join(chr(i) for i in range(33, 127) if chr(i) not in {'"', '\\'})
NODE_ID_ALPHABET = string.ascii_letters + string.digits + '_.:-'


def _catalog_schemas(state, kind, *, alphabet, fold_case=False):
    consumed = {query for action, query, _offset in state.requests if action == kind}
    branches = [action_schema(kind, {
        'query': string_excluding(consumed, alphabet=alphabet, minimum=0, fold_case=fold_case),
        'offset': {'const': 0},
    })]
    for row in state.lookup_successors(kind):
        branches.append(action_schema(kind, {
            'query': {'const': row['query']}, 'offset': {'const': row['next_offset']},
        }))
    return branches


def operation_next_schema(state, *, nodes, source_units, pinned_source, dependency_types,
                          can_complete):
    actions = [*OPERATIONS, *(['complete'] if can_complete else [])]
    branches = [obj({'action': {'enum': actions}, 'query': {'const': ''}, 'offset': {'const': 0}}),
                action_schema('extension', {'query': ID, 'offset': {'const': 0}}),
                *symbol_lookup_schemas(state), *binding_lookup_schemas(state)]
    consumed_symbols = {query for kind, query, _offset in state.requests if kind == 'symbols'}
    available_types = sorted(key for key in dependency_types
                             if symbol_query_key(key) not in consumed_symbols)
    if available_types:
        branches.append(action_schema('inspect_dependency', {
            'query': {'enum': available_types}, 'offset': {'const': 0},
        }))
    offset = state.next_declaration_offset()
    if offset is not None:
        branches.append(action_schema('declarations', {'query': {'const': ''}, 'offset': {'const': offset}}))

    if nodes:
        branches.extend(_catalog_schemas(state, 'node_index', alphabet=NODE_ID_ALPHABET))
        referenced = {ref for row in nodes for ref in row['input_refs']}
        roots = [row['node_id'] for row in nodes if row['node_id'] not in referenced]
        visible = {row['node_id'] for row in nodes[-PAGE_SIZE:]} | set(roots[:PAGE_SIZE])
        for row in state.lookup_cursors.values():
            if row['kind'] == 'node_index':
                matches = [node for node in nodes if row['query'] in node['node_id']]
                visible.update(node['node_id'] for node in matches[row['offset']:row['offset'] + PAGE_SIZE])
        unread = sorted(key for key in visible if ('nodes', key, 0) not in state.requests)
        if unread:
            # One full node per read; every older node remains discoverable by
            # the complete, pageable index. No graph-size limit is introduced.
            branches.append(action_schema('nodes', {'query': {'enum': unread}, 'offset': {'const': 0}}))

    if source_units:
        branches.extend(_catalog_schemas(state, 'sources', alphabet=_SELECTOR_ALPHABET, fold_case=True))
        source_ids = sorted(source_units)
        visible = set(source_ids[:PAGE_SIZE])
        if pinned_source in source_units:
            visible.add(pinned_source)
        for row in state.lookup_cursors.values():
            if row['kind'] == 'sources':
                matches = [key for key in source_ids if row['query'].casefold() in key.casefold()]
                visible.update(matches[row['offset']:row['offset'] + PAGE_SIZE])
            elif row['kind'] == 'source':
                visible.add(row['query'])
        for key in sorted(visible):
            consumed = {offset for kind, query, offset in state.requests
                        if kind == 'source' and query == key}
            offset = max(consumed) + 1 if consumed else 0
            if offset < len(source_units[key]):
                branches.append(action_schema('source', {'query': {'const': key}, 'offset': {'const': offset}}))
    return {'oneOf': branches}
