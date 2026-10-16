"""Explicit, scoped local bindings for multiply used forward input slots."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .construction_feedback import graph_digest
from .construction_operation_contracts import named_values, pending_input_contracts, result_fits
from .java_operation_ir import LEAVES, PRIMITIVES, parameter_records, parameter_types

VALUE_READS = LEAVES


def input_use_count(nodes, slot):
    return sum(node['input_refs'].count(slot) for node in nodes
               if node['node_id'] != slot)


def _placement(nodes, slot, declaration):
    by_id = {node['node_id']: node for node in nodes}
    if declaration['kind'] == 'FIELD' or slot in by_id or input_use_count(nodes, slot) < 2:
        return None
    referenced = {ref for node in nodes for ref in node['input_refs']}
    roots = [key for key in by_id if key not in referenced]
    occurrences = []
    stack = [(key, [('roots', '', index)], (), frozenset())
             for index, key in reversed(list(enumerate(roots)))]
    while stack:
        key, regions, earlier, ancestors = stack.pop()
        if key == slot:
            occurrences.append((regions, earlier))
            continue
        node = by_id.get(key)
        if node is None:
            continue
        if key in ancestors:
            raise ValueError('CONSTRUCTION_NODE_CYCLE')
        op, refs, attrs = node['op'], node['input_refs'], node['attributes']
        ancestors = ancestors | {key}
        for index in reversed(range(len(refs))):
            ref = refs[index]
            child_regions, child_earlier = regions, earlier
            body = False
            if op == 'block':
                child_regions = [*regions, ('block', key, index)]
                child_earlier = ()
            elif op in {'if', 'switch'} and index > 0:
                body = True
            elif op in {'while', 'do_while', 'foreach', 'for'}:
                child_regions, child_earlier = [], ()
                body = index == (3 if op == 'for' else 1)
            elif op in {'lambda', 'try'}:
                body = True
            elif op == 'synchronized':
                child_regions, child_earlier = [], ()
                body = index == 1
            elif op == 'constructor_call':
                # Java forbids a declaration before explicit this()/super().
                child_regions, child_earlier = [], ()
            elif ((op == 'binary' and attrs['operator'] in {'&&', '||'} and index == 1)
                  or (op == 'ternary' and index > 0)):
                child_regions, child_earlier = [], ()
            else:
                child_earlier = (*earlier, *(prior for prior in refs[:index]
                    if by_id.get(prior, {}).get('op') not in LEAVES))
            if body:
                # The binding stays inside this exact branch/callback/loop/
                # catch/finally/lock body; no outer region crosses the boundary.
                child_regions = [('body', key, index)]
                child_earlier = ()
            stack.append((ref, child_regions, child_earlier, ancestors))
    if len(occurrences) != input_use_count(nodes, slot):
        return None
    def region_id(region):
        kind, container, index = region
        return kind, container, index if kind == 'body' else None

    common = {region_id(region) for region in occurrences[0][0]}
    for regions, _earlier in occurrences[1:]:
        common &= {region_id(region) for region in regions}
    nearest = next((region_id(region) for region in reversed(occurrences[0][0])
                    if region_id(region) in common), None)
    if nearest is None:
        return None
    kind, container, _body_index = nearest
    index = min(position for regions, _earlier in occurrences
                for candidate, owner, position in regions
                if region_id((candidate, owner, position)) == nearest)
    before = roots[index] if kind == 'roots' else by_id[container]['input_refs'][index]
    prior = list(dict.fromkeys(ref for _regions, earlier in occurrences for ref in earlier))
    site = {'kind': kind, 'container': container, 'index': index, 'before_node_id': before}
    selector = [graph_digest(nodes), slot, site]
    site['placement_ref'] = 'shared-placement:' + hashlib.sha256(
        json.dumps(selector, sort_keys=True).encode()).hexdigest()[:24]
    site['prior_evaluation_refs'] = prior
    return site


def shared_input_choices(state, nodes, declaration):
    slot = (state.selection_goal or {}).get('node_id')
    count = input_use_count(nodes, slot)
    if count < 2 or any(node['node_id'] == slot for node in nodes):
        return None
    contracts = pending_input_contracts(nodes, declaration, state.symbols).get(slot, ())
    readable = all(row['mode'] == 'expression'
                   and not (row['operation'] == 'assign' and row['input_index'] == 0)
                   and (not row.get('allowed_operations') or 'local' in row['allowed_operations'])
                   for row in contracts)
    placement = _placement(nodes, slot, declaration) if contracts and readable else None
    # A consumer may constrain only shape (array_length), or no exact type
    # (binary operands). Preserve exact array types from inspected signatures
    # and local graph evidence instead of requiring a failed call first.
    types = set(PRIMITIVES) - {'void'}
    types.update(node['typed_result'] for node in nodes)
    for values in named_values(nodes, declaration, state.symbols, node_id=slot).values():
        types.update(type_ for _name, type_ in values)
    visible = [*state.symbol_lookup.get('matches', ()), *state.memory.visible_symbols()]
    for row in visible:
        if row.get('return_type'):
            types.add(row['return_type'])
        types.update(parameter_types(row))
        if row.get('kind') in {'CLASS', 'CONSTRUCTOR'}:
            types.add(row['owner'])
    for row in contracts:
        types.update(value for value in (row.get('expected_type'),) if value)
        types.update(row.get('expected_types', ()))
        types.update(row.get('allowed_types', ()))
        if row.get('expected_element_type'):
            types.add(row['expected_element_type'] + '[]')
    observation = state.validation_observation or {}
    candidate = observation.get('rejected_node') or {}
    if candidate.get('node_id') == slot and candidate.get('typed_result'):
        types.add(candidate['typed_result'])
    from .construction_declaration_choices import accessible_type
    types = sorted(type_ for type_ in types if type_ != 'void'
                   and (type_.split('[', 1)[0] in PRIMITIVES
                        or accessible_type(state, state.symbols.get(type_.split('[', 1)[0], {})))
                   and result_fits(type_, contracts, state.evidence_index))
    failed = (state.selection_goal or {}).get('rejected_shared_bindings', ())
    if placement:
        types = [type_ for type_ in types if not any(
            row['placement_ref'] == placement['placement_ref'] and row['type_ref'] == type_
            for row in failed)]
    return {'node_id': slot, 'use_count': count, 'placement': placement,
            'binding_types': types,
            'evaluation_policy': 'once_before_statement',
            'meaning': 'A shared input needs a leaf value or an explicit local binding. '
                       'Binding chooses one evaluation before the named statement, including before '
                       'its listed earlier subexpressions. Select it only for the intended order; '
                       'otherwise revise a consumer. No binding crosses a control boundary.'}


def bind_shared_input(nodes, slot, type_ref, placement_ref, declaration):
    """Construct one exact rewrite; callers still run the ordinary partial validator."""
    site = _placement(nodes, slot, declaration)
    if (site is None or site['placement_ref'] != placement_ref
            or not isinstance(type_ref, str) or not type_ref.strip() or type_ref == 'void'):
        raise ValueError('CONSTRUCTION_SHARED_BINDING_INVALID')
    occupied = {node['node_id'] for node in nodes}
    occupied.update(ref for node in nodes for ref in node['input_refs'])
    seed = hashlib.sha256((graph_digest(nodes) + '\0' + slot).encode()).hexdigest()[:16]

    def fresh(prefix):
        base, number = prefix + seed, 0
        key = base
        while key in occupied:
            number += 1
            key = base + '_' + str(number)
        occupied.add(key)
        return key

    names = {row['name'] for row in parameter_records(declaration)}
    for node in nodes:
        attrs = node['attributes']
        if attrs.get('name'):
            names.add(attrs['name'])
        names.update(attrs.get('parameters', ()))
        names.update(row['name'] for row in attrs.get('catches', ()))
    name, suffix = 'mmmShared' + seed, 0
    while name in names:
        suffix += 1
        name = 'mmmShared' + seed + '_' + str(suffix)
    declaration_id, initializer = fresh('shared_declaration_'), fresh('shared_initializer_')
    local = {'node_id': slot, 'op': 'local', 'input_refs': [], 'typed_result': type_ref,
             'attributes': {'name': name}}
    binding = {'node_id': declaration_id, 'op': 'declare', 'input_refs': [initializer],
               'typed_result': type_ref, 'attributes': {'name': name, 'final': True}}
    result = deepcopy(list(nodes))
    changed = set()
    if site['kind'] == 'roots':
        index = next(index for index, node in enumerate(result) if node['node_id'] == site['before_node_id'])
        result.insert(index, binding)
    else:
        parent = next(node for node in result if node['node_id'] == site['container'])
        changed.add(parent['node_id'])
        if site['kind'] == 'block':
            parent['input_refs'].insert(site['index'], declaration_id)
        else:
            wrapper = fresh('shared_scope_')
            parent['input_refs'][site['index']] = wrapper
            result.append({'node_id': wrapper, 'op': 'block', 'input_refs': [declaration_id, site['before_node_id']],
                           'typed_result': 'void', 'attributes': {}})
        result.append(binding)
    result.append(local)
    return tuple(result), {'initializer': initializer, 'declaration': declaration_id,
                           'local': slot, 'name': name, 'changed_nodes': changed}


def is_shared_binding_revision(existing, candidate, declaration):
    """A CAS edit admits only the deterministic binding of one existing hole."""
    previous = {node['node_id'] for node in existing}
    additions = [node for node in candidate if node['node_id'] not in previous and node['op'] == 'local'
                 and input_use_count(existing, node['node_id']) > 1]
    if len(additions) != 1:
        return False
    local = additions[0]
    site = _placement(existing, local['node_id'], declaration)
    if site is None:
        return False
    expected, _details = bind_shared_input(existing, local['node_id'], local['typed_result'],
                                          site['placement_ref'], declaration)
    return tuple(candidate) == expected
