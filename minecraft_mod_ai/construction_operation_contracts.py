"""Mechanical input obligations derived from the accepted partial operation graph."""
from __future__ import annotations
from .construction_type_contracts import owner_package

from .java_operation_ir import parameter_types, invocation_parameter_types


def result_fits(type_, consumers, evidence_index):
    for row in consumers:
        if row['mode'] == 'expression' and type_ == 'void':
            return False
        if row.get('expected_type') and not evidence_index.assignable(type_, row['expected_type']):
            return False
        if row.get('expected_types') and not all(evidence_index.assignable(type_, value) for value in row['expected_types']):
            return False
        if row.get('allowed_types') and type_ not in row['allowed_types']:
            return False
        if row.get('required_shape') == 'array' and not type_.endswith('[]'):
            return False
        if row.get('expected_element_type') and (not type_.endswith('[]')
                or not evidence_index.assignable(type_[:-2], row['expected_element_type'])):
            return False
    return True


def named_values(nodes, declaration, symbols, *, node_id=None):
    from .construction_lexical import names_at_slot
    # A new root follows preceding roots. An input slot has only the names
    # visible in all its actual lexical contexts, never sibling callback names.
    slot = node_id if node_id is not None else '__construction_next_root__'
    while node_id is None and any(node['node_id'] == slot or slot in node['input_refs'] for node in nodes):
        slot += '_'
    return names_at_slot(nodes, declaration, symbols, slot)


def pending_input_contracts(nodes, declaration, symbols):
    by_id = {row['node_id']: row for row in nodes}
    parents = {}
    for row in nodes:
        for ref in row['input_refs']:
            parents.setdefault(ref, []).append(row)

    def returns_for(key):
        frontier, seen = [(key, 0)], set()
        nearest, results = None, set()
        while frontier:
            current, depth = frontier.pop(0)
            if current in seen or (nearest is not None and depth > nearest):
                continue
            seen.add(current)
            for parent in parents.get(current, ()):
                if parent['op'] == 'lambda':
                    nearest = depth
                    results.add(symbols.get(parent['typed_result'], {}).get('functional_return_type', 'void'))
                else:
                    frontier.append((parent['node_id'], depth + 1))
        return sorted(results or {declaration.get('return_type', 'void')})

    result = {}
    for node in nodes:
        op, refs, attrs = node['op'], node['input_refs'], node['attributes']
        expression_indices = {
            'if': {0}, 'while': {0}, 'do_while': {0}, 'for': {1}, 'foreach': {0},
            'switch': {0}, 'synchronized': {0}, 'lambda': set(), 'block': set(), 'try': set(),
        }.get(op, set(range(len(refs))))
        api_types = None
        if op in {'call', 'field', 'new', 'constructor_call'}:
            row = symbols[attrs['symbol_ref']]
            api_types = [] if op == 'field' else invocation_parameter_types(row) if op in {'new', 'constructor_call'} else parameter_types(row)
            if op not in {'new', 'constructor_call'} and not row.get('static'):
                receiver = row['owner']
                if row.get('access', 0) & 4 and owner_package(symbols, row.get('declaring_owner') or receiver) != owner_package(symbols, declaration['owner']):
                    receiver = declaration['owner']
                api_types = [receiver, *api_types]
        for index, ref in enumerate(refs):
            if ref in by_id:
                continue
            row = {'consumer': node['node_id'], 'operation': op, 'input_index': index,
                   'mode': 'expression' if index in expression_indices else 'statement'}
            if api_types is not None and index < len(api_types):
                row['expected_type'] = api_types[index]
            elif op == 'assign' and index == 0:
                row['allowed_operations'] = ['local', 'field', 'array_get']
            elif op == 'declare':
                row['expected_type'] = node['typed_result']
            elif (op in {'if', 'while', 'do_while', 'ternary'} and index == 0) or (op == 'for' and index == 1):
                row['allowed_types'] = ['boolean']
            elif op == 'return':
                row['expected_types'] = returns_for(node['node_id'])
            elif op == 'array_get':
                row['allowed_types'] = [node['typed_result'] + '[]'] if index == 0 else ['int', 'byte', 'short', 'char']
            elif op == 'new_array':
                if attrs.get('sized'):
                    row['allowed_types'] = ['int', 'byte', 'short', 'char']
                else:
                    row['expected_type'] = attrs['element_type']
            elif op == 'ternary' and index > 0:
                row['expected_type'] = node['typed_result']
            elif op == 'foreach' and index == 0:
                row.update(required_shape='array', expected_element_type=attrs['element_type'])
            elif op == 'array_length':
                row['required_shape'] = 'array'
            elif op == 'switch' and index == 0:
                row['allowed_types'] = [attrs['selector_type']]
            elif op == 'throw':
                row['expected_type'] = 'java.lang.Throwable'
            elif op == 'for' and index in {0, 2}:
                row['allowed_operations'] = ['declare', 'assign', 'call', 'new'] if index == 0 else ['assign', 'call', 'new']
            result.setdefault(ref, []).append(row)
    return result
