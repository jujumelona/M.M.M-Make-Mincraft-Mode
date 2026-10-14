"""Bind one authored operation to real API/source evidence and signature facts."""
from __future__ import annotations
from .construction_type_contracts import owner_package

import hashlib

from .construction_author import ask, obj, operation_schema
from .construction_declaration_choices import action_schema, choice_actions, type_candidates
from .construction_operation_choices import operation_context_branches
from .construction_state import symbol_record, RECENT_SIZE
from .java_operation_ir import parameter_types, invocation_parameter_types
from .construction_operation_contracts import named_values, pending_input_contracts, result_fits

API_KINDS = {'call': 'METHOD', 'new': 'CONSTRUCTOR', 'field': 'FIELD', 'constructor_call': 'CONSTRUCTOR'}
TYPE_BOUND_OPERATIONS = frozenset({'declare', 'literal', 'cast', 'ternary', 'array_get', 'new_array'})


def available_types(state, nodes, declaration, requirements):
    types = set(type_candidates(state))
    types.update(row['typed_result'] for row in nodes)
    types.update(parameter_types(declaration))
    types.add(declaration['return_type'])
    types.update(row['expected_type'] for row in requirements if row.get('expected_type'))
    types.update(row['expected_element_type'] for row in requirements if row.get('expected_element_type'))
    types.update(value for row in requirements for value in row.get('expected_types', ()))
    types.update(value for row in requirements for value in row.get('allowed_types', ()))
    for name in ('java.lang.String', 'java.lang.Object'):
        if name in state.symbols:
            types.add(name)
    return types


def type_binding_branches(state, types, requirements, op):
    branches = []
    bases = sorted({type_.split('[', 1)[0] for type_ in types} - {'void'})
    for base in bases:
        # JVMS array dimensions are structural syntax, not an inventory/name
        # lookup. Keep every valid dimension without materializing huge enums.
        minimum, maximum = (1, 255) if op == 'new_array' else (0, 254 if op == 'array_get' else 255)
        dimensions = [dimension for dimension in range(minimum, maximum + 1)
                      if result_fits(base + '[]' * dimension, requirements, state.evidence_index)]
        if not dimensions:
            continue
        if dimensions == list(range(dimensions[0], dimensions[-1] + 1)):
            shape = {'type': 'integer', 'minimum': dimensions[0], 'maximum': dimensions[-1]}
        else:
            shape = {'enum': dimensions}
        properties = {'base_type': {'const': base}, 'array_dimensions': shape}
        if op == 'new_array':
            properties['sized'] = {'type': 'boolean'}
        branches.append(action_schema('write_type', properties))
    return branches


def bind_operation(router, state, *, goal, payload, nodes, source_units, pinned_source,
                   read_lookup, instruction, save_author):
    op = goal['action']
    kinds = (API_KINDS[op],) if op in API_KINDS else ('CLASS',)
    requirements = pending_input_contracts(nodes, payload['declaration'], state.symbols).get(goal['node_id'], ())

    def fits(type_):
        return result_fits(type_, requirements, state.evidence_index)

    purpose = '\n'.join(str(value) for value in (
        goal['purpose'], payload['intent'], payload['effects'],
        payload['declaration'].get('purpose', ''),
        *[row.get('expected_type', '') for row in requirements],
        *[row.get('expected_element_type', '') for row in requirements],
        *payload['state_reads'], *payload['state_writes'],
    ) if value)
    state.prepare_symbols(purpose, kinds=kinds, preserve_scope=True)
    save_author()
    while True:
        evidence = state.symbol_lookup
        matches = [row for row in evidence.get('matches', ()) if row['kind'] in kinds
                   and (op not in API_KINDS or fits(row['owner'] if op == 'new' else row['return_type']))]
        if op in API_KINDS:
            local_keys = set(state.declaration_ids[-RECENT_SIZE:])
            if state.lookup.get('kind') == 'declarations':
                local_keys.update(row['symbol_id'] for row in state.lookup.get('declarations', ()))
            shown = {row['symbol_id'] for row in matches}
            for key in sorted(local_keys - shown):
                row = state.symbols[key]
                if (row['kind'] in kinds and state.evidence_index.accessible(key, constructor=op == 'new')
                        and fits(row['owner'] if op == 'new' else row['return_type'])):
                    matches.append(symbol_record(key, row))
            if op == 'constructor_call':
                declaration = payload['declaration']
                parent = declaration.get('class_superclass') or state.symbols[state.owner].get('superclass') or 'java.lang.Object'
                matches = [row for row in matches if row['owner'] in {state.owner, parent}
                           and row['symbol_id'] != declaration['symbol_id']]
                # The selected header and own declaration set already prove
                # these exact constructor owners; no guessed-name search.
                shown = {row['symbol_id'] for row in matches}
                for key, row in state.symbols.items():
                    if (key not in shown and key != declaration['symbol_id'] and row.get('kind') == 'CONSTRUCTOR'
                            and row['owner'] in {state.owner, parent}
                            and state.evidence_index.accessible(key)):
                        matches.append(symbol_record(key, row))
        if kinds == ('CLASS',):
            # A previously selected API's input signature is already actual
            # type evidence, including callback interfaces not on this page.
            required_types = {row['expected_type'].split('[', 1)[0] for row in requirements if row.get('expected_type')}
            required_types.update(value.split('[', 1)[0] for row in requirements for value in row.get('expected_types', ()))
            shown = {row['symbol_id'] for row in matches}
            matches.extend(symbol_record(key, state.symbols[key]) for key in sorted(required_types - shown)
                           if state.symbols.get(key, {}).get('kind') == 'CLASS' and state.evidence_index.accessible(key))
        branches = operation_context_branches(state, nodes=nodes, source_units=source_units,
                                             pinned_source=pinned_source)
        if op in API_KINDS:
            if matches:
                branches.insert(0, action_schema('api', {'symbol_ref': {'enum': [row['symbol_id'] for row in matches]}}))
        elif op == 'lambda':
            functional = [row['symbol_id'] for row in matches
                          if 'functional_parameters' in row and fits(row['symbol_id'])]
            if functional:
                branches.insert(0, action_schema('functional_type', {'type_ref': {'enum': functional}}))
        elif op in TYPE_BOUND_OPERATIONS:
            branches[0:0] = type_binding_branches(state,
                available_types(state, nodes, payload['declaration'], requirements), requirements, op)
        else:
            branches.insert(0, action_schema('write'))
        if not branches:
            raise ValueError('CONSTRUCTION_OPERATION_EVIDENCE_MISSING: ' + op)
        schema = {'oneOf': branches}
        current = {**payload, 'operation_goal': goal,
                   **state.payload(choice_actions(schema), **{key: payload['construction_state'][key]
                       for key in ('current_member', 'graph', 'sources') if key in payload['construction_state']}),
                   'api_candidates': matches,
                   'required_by_consumers': requirements}
        choice = ask(router, 'construction_operation_bind', current, schema,
            'Resolve only this selected operation. Select the real API whose behavior and signature '
            'fit operation_goal, or read a returned source/type/configuration pointer to resolve a '
            'specific missing fact. scopes selects a real package/type branch by meaning when '
            'the right API is not on the current page. API names are returned candidates, not search guesses. '
            'When this operation can be written, select api, write_type, functional_type or write now. '
            'write_type binds an actual base type plus array dimensions; new_array also binds sized/initializer form. '
            'Another operation is chosen '
            'only after this node is written. ' + instruction)
        if choice['action'] in {'api', 'write', 'write_type', 'functional_type'}:
            break
        read_lookup(choice)
        save_author()
    schema = operation_schema(op, node_ids=(row['node_id'] for row in nodes))
    schema['properties']['node_id'] = {'const': goal['node_id']}
    from .construction_declaration_choices import string_excluding
    from .construction_operation_choices import NODE_ID_ALPHABET
    schema['properties']['input_refs']['items'] = string_excluding((goal['node_id'],), alphabet=NODE_ID_ALPHABET)
    types = available_types(state, nodes, payload['declaration'], requirements)
    selected_type = None
    if op in TYPE_BOUND_OPERATIONS:
        selected_type = choice['base_type'] + '[]' * choice['array_dimensions']
        types.add(selected_type)
    if op in API_KINDS:
        key = choice['symbol_ref']
        row = state.symbols[key]
        inputs = [] if op == 'field' else invocation_parameter_types(row) if op in {'new', 'constructor_call'} else parameter_types(row)
        if op not in {'new', 'constructor_call'} and not row.get('static'):
            receiver = row['owner']
            if row.get('access', 0) & 4 and owner_package(state.symbols, row['owner']) != owner_package(state.symbols, state.owner):
                receiver = state.owner
            inputs = [receiver, *inputs]
        schema['properties']['attributes']['properties']['symbol_ref'] = {'const': key}
        schema['properties']['typed_result'] = {'const': row['owner'] if op == 'new' else 'void' if op == 'constructor_call' else row['return_type']}
        if op == 'constructor_call':
            schema['properties']['attributes']['properties']['target'] = {'const': 'this' if row['owner'] == state.owner else 'super'}
        schema['properties']['input_refs']['minItems'] = len(inputs)
        schema['properties']['input_refs']['maxItems'] = len(inputs)
        current['selected_api'] = symbol_record(key, row)
        current['required_input_types'] = inputs
    else:
        nonvoid = sorted(types - {'void'})
        result_types = sorted(type_ for type_ in types if fits(type_))
        # Statement nodes' void result describes syntax; declarations still
        # carry the actual local type. Consumer expression contracts constrain
        # only value-producing nodes.
        schema['properties']['typed_result'] = {'enum': result_types}
        attrs = schema['properties']['attributes']['properties']
        operators = {
            'binary': ['==', '!=', '&&', '||', '+', '-', '*', '/', '%', '<', '>', '<=', '>=', '&', '|', '^', '<<', '>>', '>>>'],
            'unary': ['!', '+', '-', '~'],
            'assign': ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='],
        }
        if op in operators:
            attrs['operator'] = {'enum': operators[op]}
        if op in {'literal', 'parameter', 'local', 'declare', 'binary', 'unary', 'assign',
                  'cast', 'array_get', 'new_array', 'ternary', 'lambda'}:
            result_types = [type_ for type_ in result_types if type_ != 'void']
            schema['properties']['typed_result'] = {'enum': result_types}
        if selected_type is not None:
            result_types = [selected_type]
            schema['properties']['typed_result'] = {'const': selected_type}
        for field in ('type_ref', 'element_type'):
            if field in attrs:
                attrs[field] = {'enum': nonvoid}
        if op in {'block', 'return', 'if', 'while', 'for', 'do_while', 'foreach', 'switch',
                  'throw', 'try', 'synchronized', 'break', 'continue'}:
            schema['properties']['typed_result'] = {'const': 'void'}
        elif op == 'this':
            schema['properties']['typed_result'] = {'const': state.owner}
        elif op == 'array_length':
            schema['properties']['typed_result'] = {'const': 'int'}
        elif op == 'instanceof':
            schema['properties']['typed_result'] = {'const': 'boolean'}
        elif op == 'lambda':
            key = choice['type_ref']
            row = state.symbols[key]
            schema['properties']['typed_result'] = {'const': key}
            schema['properties']['input_refs'].update(minItems=1, maxItems=1)
            prefix = 'lambda_' + hashlib.sha256(goal['node_id'].encode()).hexdigest()[:16] + '_arg'
            attrs['parameters'] = {'const': [prefix + str(index) for index in range(len(row['functional_parameters']))]}
            current['functional_signature'] = symbol_record(key, row)
        if op in {'literal', 'parameter', 'local', 'this', 'break', 'continue'}:
            schema['properties']['input_refs'].update(minItems=0, maxItems=0)
        if op in {'parameter', 'local'}:
            values = named_values(nodes, payload['declaration'], state.symbols)[op]
            variants = []
            from copy import deepcopy
            for name, type_ in values:
                if not fits(type_):
                    continue
                branch = deepcopy(schema)
                branch['properties']['typed_result'] = {'const': type_}
                branch['properties']['attributes']['properties']['name'] = {'const': name}
                variants.append(branch)
            if not variants:
                raise ValueError('CONSTRUCTION_OPERATION_NAMED_VALUE_MISSING: ' + op)
            schema = {'oneOf': variants}
        if op == 'cast':
            from copy import deepcopy
            variants = []
            for type_ in result_types:
                if type_ == 'void':
                    continue
                branch = deepcopy(schema)
                branch['properties']['typed_result'] = {'const': type_}
                branch['properties']['attributes']['properties']['type_ref'] = {'const': type_}
                variants.append(branch)
            schema = {'oneOf': variants}
        if op == 'new_array':
            sized = choice['sized']
            attrs['sized'] = {'const': sized}
            attrs['element_type'] = {'const': choice['base_type'] if sized else selected_type[:-2]}
            if sized:
                schema['properties']['input_refs'].update(minItems=choice['array_dimensions'], maxItems=choice['array_dimensions'])
        if op == 'literal':
            from copy import deepcopy
            variants = []
            for type_ in result_types:
                value = None
                if type_ == 'boolean':
                    value = {'type': 'boolean'}
                elif type_ in {'byte', 'short', 'int', 'long'}:
                    bits = {'byte': 8, 'short': 16, 'int': 32, 'long': 64}[type_]
                    value = {'type': 'integer', 'minimum': -(2 ** (bits - 1)), 'maximum': 2 ** (bits - 1) - 1}
                elif type_ in {'float', 'double'}:
                    value = {'type': 'number'}
                elif type_ == 'char':
                    value = {'type': 'string', 'minLength': 1, 'maxLength': 1}
                elif type_ == 'java.lang.String':
                    value = {'anyOf': [{'type': 'string', 'maxLength': 240}, {'type': 'null'}]}
                elif type_ != 'void':
                    value = {'type': 'null'}
                if value is not None:
                    branch = deepcopy(schema)
                    branch['properties']['typed_result'] = {'const': type_}
                    branch['properties']['attributes']['properties']['value'] = value
                    variants.append(branch)
            schema = {'oneOf': variants}
        arities = {'binary': (2, 2), 'unary': (1, 1), 'assign': (2, 2), 'cast': (1, 1),
                   'instanceof': (1, 1), 'array_get': (2, 2), 'array_length': (1, 1),
                   'ternary': (3, 3), 'declare': (0, 1), 'return': (0, 1), 'if': (2, 3),
                   'while': (2, 2), 'do_while': (2, 2), 'for': (4, 4), 'foreach': (2, 2),
                   'throw': (1, 1), 'synchronized': (2, 2)}
        if op in {'switch', 'try'}:
            if op == 'switch':
                selector_types = ['byte', 'short', 'char', 'int']
                if 'java.lang.String' in state.symbols:
                    selector_types.append('java.lang.String')
                selector = ask(router, 'construction_switch_selector_type', current,
                    obj({'selector_type': {'enum': selector_types}}),
                    'Choose the actual switch selector type; labels and its input are bound to that type. ' + instruction)['selector_type']
                if selector == 'java.lang.String':
                    literal = {'type': 'string'}
                elif selector == 'char':
                    literal = {'type': 'string', 'minLength': 1, 'maxLength': 1}
                else:
                    bits = {'byte': 8, 'short': 16, 'int': 32}[selector]
                    literal = {'type': 'integer', 'minimum': -(2 ** (bits - 1)), 'maximum': 2 ** (bits - 1) - 1}
                attribute_schema = obj({'selector_type': {'const': selector}, 'cases': {
                    'type': 'array', 'items': {'oneOf': [obj({'kind': {'const': 'default'}}),
                        obj({'kind': {'const': 'value'}, 'value': literal})]}}})
            else:
                throwable_types = sorted(key for key, row in state.symbols.items()
                    if row.get('kind') == 'CLASS' and state.evidence_index.accessible(key)
                    and state.evidence_index._subtype(key, 'java.lang.Throwable'))
                catches = {'type': 'array', 'minItems': 1, 'items': obj({
                    'name': attrs['catches']['items']['properties']['name'],
                    'type_ref': {'enum': throwable_types}})}
                attribute_schema = {'oneOf': [obj({'catches': catches, 'finally': {'type': 'boolean'}}),
                    obj({'catches': {'const': []}, 'finally': {'const': True}})]}
            control_attributes = ask(router, 'construction_' + op + '_structure', current, attribute_schema,
                'Choose this control node structure. Each listed case/catch creates one required body slot. '
                'Default is a label kind, distinct from any string value. ' + instruction)
            schema['properties']['attributes'] = {'const': control_attributes}
            count = (1 + len(control_attributes['cases']) if op == 'switch' else
                     1 + len(control_attributes['catches']) + bool(control_attributes['finally']))
            arities[op] = (count, count)
            current['control_structure'] = control_attributes
        if op in arities:
            minimum, maximum = arities[op]
            for branch in schema.get('oneOf', [schema]):
                branch['properties']['input_refs'].update(minItems=minimum, maxItems=maximum)
    return dict(ask(router, 'construction_operation_' + op, current, schema, instruction))
