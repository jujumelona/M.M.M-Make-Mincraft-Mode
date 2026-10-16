"""Bind one authored operation to real API/source evidence and signature facts."""
from __future__ import annotations
from .construction_type_contracts import owner_package

import hashlib
import json

from .construction_author import ID, ask, obj, operation_schema
from .construction_declaration_choices import action_schema, choice_actions, type_candidates
from .construction_operation_choices import (operation_context_branches, MAX_OPERATION_EVIDENCE_READS,
                                           dependency_revision_schema, dependency_revision_choices)
from .construction_state import symbol_record, RECENT_SIZE, RESELECTABLE_OPERATIONS
from .java_operation_ir import parameter_types, invocation_parameter_types, LEAVES
from .construction_shared_inputs import input_use_count, VALUE_READS
from .construction_operation_contracts import named_values, pending_input_contracts, result_fits

API_KINDS = {'call': 'METHOD', 'new': 'CONSTRUCTOR', 'field': 'FIELD', 'constructor_call': 'CONSTRUCTOR'}
TYPE_BOUND_OPERATIONS = frozenset({'declare', 'literal', 'cast', 'ternary', 'array_get', 'new_array'})
SYMBOL_BOUND_OPERATIONS = frozenset(API_KINDS) | TYPE_BOUND_OPERATIONS | {
    'lambda', 'instanceof', 'foreach', 'switch', 'try',
}


def _requires_missing_self_receiver(row, requirements, evidence_index):
    """Returning a receiver's type does not construct the required receiver."""
    if not requirements or row.get('static') or row.get('access', 0) & 8:
        return False
    return (evidence_index.assignable(row.get('return_type', 'void'), row['owner'])
            and result_fits(row['owner'], requirements, evidence_index))


def _automatic_literal_type(op, requirements, branches):
    """Bind a unique scalar type without wasting a model completion."""
    if op != 'literal' or not requirements:
        return None
    options = [branch['properties'] for branch in branches
               if branch['properties'].get('action', {}).get('const') == 'write_type']
    if len(options) != 1:
        return None
    dim = options[0]['array_dimensions']
    if not (dim.get('enum') == [0] or
            (dim.get('minimum') == 0 and dim.get('maximum') == 0)):
        return None
    return {'action': 'write_type', 'base_type': options[0]['base_type']['const'],
            'array_dimensions': 0}


def _usable_api_candidate(key, row, *, op, kinds, current_member, rejected_symbols, state, fits,
                          requirements=()):
    # A declaration whose body is being authored is an obligation, not
    # existing implementation evidence. Keep sibling helpers available.
    if op == 'call' and (key == current_member or row.get('inherited_from') == current_member):
        return False
    if key in rejected_symbols or row.get('inherited_from') in rejected_symbols:
        return False
    if op == 'call' and _requires_missing_self_receiver(row, requirements, state.evidence_index):
        return False
    return (row['kind'] in kinds
            and state.evidence_index.accessible(key, constructor=op == 'new')
            and (op != 'call' or state.evidence_index.intent_compatible(key))
            and (op not in API_KINDS or fits(row['owner'] if op == 'new' else row['return_type'])))


def forbidden_input_references(nodes, goal_id, *, declaration=None, symbols=None):
    """Exclude all consumers of the missing slot, not just the slot itself.

    Forward references are allowed, but an existing node which transitively
    depends on the selected hole cannot also become that hole's input.
    """
    consumers = {}
    for node in nodes:
        for ref in node['input_refs']:
            consumers.setdefault(ref, set()).add(node['node_id'])
    blocked = {goal_id}
    pending = [goal_id]
    while pending:
        for user in consumers.get(pending.pop(), ()):
            if user not in blocked:
                blocked.add(user)
                pending.append(user)
    used = {ref for node in nodes if node['node_id'] != goal_id for ref in node['input_refs']}
    blocked.update(node['node_id'] for node in nodes
                   if node['op'] not in LEAVES and node['node_id'] in used)
    if declaration is not None:
        from .construction_lexical import lexical_sites
        visible = None
        for key, env, _writing, _context in lexical_sites(nodes, declaration, symbols, slot=goal_id):
            if key != goal_id:
                continue
            bindings = {(name, value[0], value[1], value[3]) for name, value in env.items()}
            visible = bindings if visible is None else visible & bindings
        values = {(name, type_, kind) for name, type_, kind, _identity in visible or ()}
        # Input roles decide definite assignment. An uninitialized local is a
        # valid assignment target, while a name absent from this scope is not.
        blocked.update(node['node_id'] for node in nodes
                       if node['op'] in {'local', 'parameter'}
                       and (node['attributes']['name'], node['typed_result'], node['op']) not in values)
    return blocked


def available_types(state, nodes, declaration, requirements):
    types = set(type_candidates(state))
    selected = state.operation_draft.get('binding_choice', {}).get('base_type')
    if selected in state.symbols and state.evidence_index.accessible(selected):
        types.add(selected)
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
    if input_use_count(nodes, goal['node_id']) > 1 and op not in VALUE_READS:
        # A stale operation draft cannot bypass the current slot's shape.
        return {'action': 'resolve_shared_input'}
    kinds = (API_KINDS[op],) if op in API_KINDS else ('CLASS',)
    requirements = pending_input_contracts(nodes, payload['declaration'], state.symbols,
                                           replacing=goal['node_id']).get(goal['node_id'], ())
    current_member = payload['declaration']['symbol_id']
    rejected_symbols = set((state.selection_goal or {}).get('rejected_symbol_refs', ()))
    rejected_symbols.update(state.rejected_apis(goal['node_id']))

    def fits(type_):
        return result_fits(type_, requirements, state.evidence_index)

    from functools import partial
    usable_api = partial(_usable_api_candidate, op=op, kinds=kinds,
                         current_member=current_member, rejected_symbols=rejected_symbols,
                         state=state, fits=fits, requirements=requirements)

    # Retrieval must describe the specific Java dependency, not the full game
    # story. Narrative terms such as "fight aliens / colonize planets" diluted
    # registration-method ranking in the production trace.
    purpose = '\n'.join(dict.fromkeys(str(value) for value in (
        goal['purpose'],
        *[row.get('expected_type', '') for row in requirements],
        *[row.get('expected_element_type', '') for row in requirements],
        *[value for row in requirements for value in row.get('expected_types', ())],
        *[value for row in requirements for value in row.get('allowed_types', ())],
    ) if value))
    # A literal is a typed value, not an API lookup. The prior search paged
    # 33,169 unrelated classes before writing a java.lang.String literal.
    resolve_symbols = op in SYMBOL_BOUND_OPERATIONS and op != 'literal'
    preferred_field = (state.selection_goal or {}).get('preferred_api') if op == 'field' else None
    if preferred_field in state.symbols:
        resolve_symbols = False

    def prepare_selection():
        rejected_symbols.update(state.rejected_apis(goal['node_id']))
        if not resolve_symbols:
            return
        selection = None
        if op in API_KINDS:
            fields = {'mode', 'expected_type', 'expected_types', 'allowed_types',
                      'expected_element_type', 'required_shape'}
            selection = {'contracts': [{key: value for key, value in row.items() if key in fields}
                                       for row in requirements],
                         'excluded_symbols': sorted(rejected_symbols | ({current_member} if op == 'call' else set())),
                         'constructor_result': op == 'new'}
        # Apply the real slot contract before selecting representatives or
        # scopes. Filtering an incompatible page afterwards made a healthy
        # classpath look empty to the small author.
        state.prepare_symbols(purpose, kinds=kinds, preserve_scope=True, selection=selection)

    prepare_selection()
    save_author()
    draft = state.operation_draft
    saved_binding = draft.get('binding_choice', {})
    saved_key = saved_binding.get('symbol_ref')
    if (saved_binding.get('action') == 'api' and saved_key in state.symbols
            and not usable_api(saved_key, state.symbols[saved_key])):
        draft.pop('binding_choice')
        save_author()
    while True:
        evidence = state.symbol_lookup
        visible = {row['symbol_id']: row for row in [*evidence.get('matches', ()), *state.memory.visible_symbols()]}
        selected = draft.get('binding_choice', {}).get('symbol_ref') or draft.get('binding_choice', {}).get('type_ref')
        if selected in state.symbols:
            visible[selected] = symbol_record(selected, state.symbols[selected])
        if preferred_field in state.symbols:
            visible[preferred_field] = symbol_record(preferred_field, state.symbols[preferred_field])
        matches = [row for row in visible.values() if usable_api(row['symbol_id'], row)]
        if op in API_KINDS:
            local_keys = set(state.declaration_ids[-RECENT_SIZE:])
            if state.lookup.get('kind') == 'declarations':
                local_keys.update(row['symbol_id'] for row in state.lookup.get('declarations', ()))
            shown = {row['symbol_id'] for row in matches}
            for key in sorted(local_keys - shown):
                row = state.symbols[key]
                if usable_api(key, row):
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
                            and usable_api(key, row)):
                        matches.append(symbol_record(key, row))
        if kinds == ('CLASS',):
            # A previously selected API's input signature is already actual
            # type evidence, including callback interfaces not on this page.
            required_types = {row['expected_type'].split('[', 1)[0] for row in requirements if row.get('expected_type')}
            required_types.update(value.split('[', 1)[0] for row in requirements for value in row.get('expected_types', ()))
            shown = {row['symbol_id'] for row in matches}
            matches.extend(symbol_record(key, state.symbols[key]) for key in sorted(required_types - shown)
                           if state.symbols.get(key, {}).get('kind') == 'CLASS' and state.evidence_index.accessible(key))
        if op in API_KINDS and not matches and max(len(state.requests), state.operation_reads) >= MAX_OPERATION_EVIDENCE_READS:
            # No verified API found in this evidence window. Return to the
            # operation selector instead of reading 484,405 candidates forever.
            # The author checkpoints this rejection and tries another operation.
            return None
        branches = operation_context_branches(state, nodes=nodes, source_units=source_units,
                                             pinned_source=pinned_source,
                                             resolve_symbols=resolve_symbols)
        branches.extend(dependency_revision_schema(state, nodes))
        if op in API_KINDS:
            if matches:
                candidates = [row['symbol_id'] for row in matches]
                branches.insert(0, action_schema('api', {'symbol_ref': {'enum': candidates}}))
                # Reject just the incompatible symbol; a different callable
                # remains eligible for the same call/new/field operation.
                branches.append(action_schema('reject_api', {
                    'symbol_ref': {'enum': candidates},
                    'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240},
                }))
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
        if op in RESELECTABLE_OPERATIONS:
            branches.append(action_schema('compose_locally', {'reason': ID}))
        if not branches:
            # A type/callback-specific choice without verified evidence must
            # return to operation selection instead of failing the entire mod.
            if op == 'lambda' or op in TYPE_BOUND_OPERATIONS:
                return None
            raise ValueError('CONSTRUCTION_OPERATION_EVIDENCE_MISSING: ' + op)
        schema = {'oneOf': branches}
        if op in API_KINDS:
            print('CONSTRUCTION API CANDIDATES: ' + json.dumps({
                'node_id': goal['node_id'], 'operation': op,
                'current_member_excluded': current_member if op == 'call' else None,
                'candidate_count': len(matches),
                'selected_page_names': [row['symbol_id'] for row in matches[:8]],
                'evidence_reads': max(len(state.requests), state.operation_reads),
            }, ensure_ascii=False), flush=True)
        current = {**payload, 'operation_goal': goal,
                   **state.payload(choice_actions(schema), **{key: payload['construction_state'][key]
                       for key in ('current_member', 'graph', 'sources') if key in payload['construction_state']}),
                   'api_candidates': matches,
                   'provisional_parent_strategies': dependency_revision_choices(state, nodes),
                   'required_by_consumers': requirements}
        choice = (draft.get('binding_choice')
                  or ({'action': 'api', 'symbol_ref': preferred_field}
                      if preferred_field in {row['symbol_id'] for row in matches} else None)
                  or _automatic_literal_type(op, requirements, branches)
                  or ask(router, 'construction_operation_bind', current, schema,
            'Resolve only this selected operation. Select the real API whose behavior and signature '
            'fit operation_goal, or read a returned source/type/configuration pointer to resolve a '
            'specific missing fact. scopes selects a real package/type branch by meaning when '
            'the right API is not on the current page. API names are returned candidates, not search guesses. '
            'When this operation can be written, select api, write_type, functional_type or write now. '
            'Do not search for a ready-made implementation of custom gameplay. If this selected '
            'candidate is semantically incorrect or its input cannot be constructed, reject_api '
            'rejects only that exact symbol; continue with other actual candidates. If the entire '
            'abstraction is unsuitable, compose_locally returns the same missing slot to Java '
            'operation selection. Explain the missing behavior and decompose it into concrete '
            'smaller steps using inspected APIs and supported local operations. This does not '
            'prove an API is absent and does not implement the slot. Never substitute a null, '
            'constant success or empty block for the required effect. '
            'write_type binds an actual base type plus array dimensions; new_array also binds sized/initializer form. '
            'If the parent API creates an input which does not serve the requested behavior or '
            'cannot be obtained here, reconsider_parent selects that actual parent for replacement. '
            'Partial admission is provisional; it does not prove its inputs are constructible. '
            'Outside compose_locally or reconsider_parent, another operation is chosen '
            'only after this node is written. ' + instruction))
        if choice['action'] == 'reject_api':
            from .fixed_template_generation import _validate_native_arguments
            _validate_native_arguments(choice, schema)
            state.reject_api(goal['node_id'], choice['symbol_ref'], choice['reason'])
            prepare_selection()
            save_author()
            continue
        if choice['action'] == 'compose_locally':
            from .fixed_template_generation import _validate_native_arguments
            _validate_native_arguments(choice, schema)
            state.reject_operation(goal, choice['reason'])
            save_author()
            return None
        if choice['action'] == 'reconsider_parent':
            from .fixed_template_generation import _validate_native_arguments
            _validate_native_arguments(choice, schema)
            return dict(choice)
        if choice['action'] in {'api', 'write', 'write_type', 'functional_type'}:
            from .fixed_template_generation import _validate_native_arguments
            _validate_native_arguments(choice, schema)
            draft['binding_choice'] = dict(choice)
            save_author()
            break
        read_lookup(choice)
        save_author()
    schema = operation_schema(op, node_ids=(row['node_id'] for row in nodes))
    schema['properties']['node_id'] = {'const': goal['node_id']}
    from .construction_declaration_choices import string_excluding
    from .construction_operation_choices import NODE_ID_ALPHABET
    schema['properties']['input_refs']['items'] = string_excluding(
        forbidden_input_references(nodes, goal['node_id'], declaration=payload['declaration'],
                                   symbols=state.symbols), alphabet=NODE_ID_ALPHABET)
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
            if row.get('access', 0) & 4 and owner_package(state.symbols, row.get('declaring_owner') or row['owner']) != owner_package(state.symbols, state.owner):
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
        from .construction_lexical import declared_names_at_slot
        from .construction_declaration_choices import identifier_schema
        declared_names = declared_names_at_slot(nodes, payload['declaration'], state.symbols, goal['node_id'])
        if op in {'declare', 'foreach'}:
            attrs['name'] = identifier_schema(declared_names)
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
            while any(prefix + str(index) in declared_names for index in range(len(row['functional_parameters']))):
                prefix += '_'
            attrs['parameters'] = {'const': [prefix + str(index) for index in range(len(row['functional_parameters']))]}
            current['functional_signature'] = symbol_record(key, row)
        if op in {'literal', 'parameter', 'local', 'this', 'break', 'continue'}:
            schema['properties']['input_refs'].update(minItems=0, maxItems=0)
        if op in {'parameter', 'local'}:
            values = named_values(nodes, payload['declaration'], state.symbols, node_id=goal['node_id'])[op]
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
            from .construction_sequence import sequence_item, unused_integer_labels
            sequence_options = dict(nodes=nodes, source_units=source_units, pinned_source=pinned_source,
                read_lookup=read_lookup, save_author=save_author, instruction=instruction,
                resolve_symbols=resolve_symbols)
            if op == 'switch':
                selector_types = ['byte', 'short', 'char', 'int']
                if 'java.lang.String' in state.symbols:
                    selector_types.append('java.lang.String')
                selector = draft.get('selector_type')
                if selector is None:
                    selector = sequence_item(router, state, 'construction_switch_selector_type', current,
                        obj({'selector_type': {'enum': selector_types}}), [], **sequence_options)['selector_type']
                    draft['selector_type'] = selector
                    save_author()
                if selector not in selector_types:
                    raise ValueError('CONSTRUCTION_SWITCH_TYPE')
                if selector == 'java.lang.String':
                    literal = {'type': 'string'}
                elif selector == 'char':
                    from .native_string_exclusions import BMP_SCALAR_PATTERN
                    literal = {'type': 'string', 'minLength': 1, 'maxLength': 1, 'pattern': BMP_SCALAR_PATTERN}
                else:
                    bits = {'byte': 8, 'short': 16, 'int': 32}[selector]
                    literal = {'type': 'integer', 'minimum': -(2 ** (bits - 1)), 'maximum': 2 ** (bits - 1) - 1}
                labels = draft.setdefault('labels', [])
                while not draft.get('labels_finished'):
                    available_literal = ({**literal, 'not': {'enum':
                        [row['value'] for row in labels if row['kind'] == 'value']}}
                        if selector in {'char', 'java.lang.String'} else
                        unused_integer_labels(literal['minimum'], literal['maximum'],
                                              [row['value'] for row in labels if row['kind'] == 'value']))
                    branches = [action_schema('finish_labels')]
                    if available_literal is not None:
                        branches.append(action_schema('value_label', {'value': available_literal}))
                    if not any(label['kind'] == 'default' for label in labels):
                        branches.append(action_schema('default_label'))
                    label = sequence_item(router, state, 'construction_switch_label_next',
                        {**current, 'selector_type': selector, 'label_count': len(labels), 'recent_labels': labels[-RECENT_SIZE:]},
                        {'oneOf': branches}, labels, **sequence_options)
                    if label['action'] == 'finish_labels':
                        draft['labels_finished'] = True
                        save_author()
                        break
                    labels.append({'kind': 'default'} if label['action'] == 'default_label' else {'kind': 'value', 'value': label['value']})
                    save_author()
                control_attributes = {'selector_type': selector, 'cases': labels}
            else:
                throwable_types = sorted(key for key, row in state.symbols.items()
                    if row.get('kind') == 'CLASS' and state.evidence_index.accessible(key)
                    and state.evidence_index._subtype(key, 'java.lang.Throwable'))
                catches = draft.setdefault('catches', [])
                while not draft.get('catches_finished'):
                    available = [type_ for type_ in throwable_types if not any(
                        state.evidence_index._subtype(type_, catch['type_ref']) for catch in catches)]
                    branches = [action_schema('finish_catches', {'finally': {'type': 'boolean'} if catches else {'const': True}})]
                    if available:
                        branches.append(action_schema('catch', {
                            'name': identifier_schema(declared_names), 'type_ref': {'enum': available}}))
                    catch = sequence_item(router, state, 'construction_try_catch_next',
                        {**current, 'catch_count': len(catches), 'recent_catches': catches[-RECENT_SIZE:]},
                        {'oneOf': branches}, catches, **sequence_options)
                    if catch['action'] == 'finish_catches':
                        draft['catches_finished'], draft['finally'] = True, catch['finally']
                        save_author()
                        break
                    catches.append({'name': catch['name'], 'type_ref': catch['type_ref']})
                    save_author()
                control_attributes = {'catches': catches, 'finally': draft['finally']}
            schema['properties']['attributes'] = {'const': control_attributes}
            count = (1 + len(control_attributes['cases']) if op == 'switch' else
                     1 + len(control_attributes['catches']) + bool(control_attributes['finally']))
            arities[op] = (count, count)
            current['control_structure'] = control_attributes
        if op in arities:
            minimum, maximum = arities[op]
            for branch in schema.get('oneOf', [schema]):
                branch['properties']['input_refs'].update(minItems=minimum, maxItems=maximum)
        if op == 'block' or (op == 'new_array' and not choice['sized']):
            from .construction_sequence import sequence_item
            references = draft.setdefault('input_refs', [])
            while not draft.get('inputs_finished'):
                input_branches = [action_schema('input', {
                    'node_ref': schema['properties']['input_refs']['items'], 'purpose': ID})]
                if references or op != 'block' or not state.operation_rejections:
                    input_branches.append(action_schema('finish_inputs'))
                reference = sequence_item(router, state, 'construction_' + op + '_input_next',
                    {**current, 'input_count': len(references), 'recent_input_refs': references[-RECENT_SIZE:]},
                    {'oneOf': input_branches},
                    references, nodes=nodes, source_units=source_units, pinned_source=pinned_source,
                    read_lookup=read_lookup, save_author=save_author,
                    instruction='Choose one actual or forward input node reference in execution order. '
                    'Give that child its own concrete implementation purpose, not the whole gameplay '
                    'description. Decompose rejected API searches into smaller executable steps. '
                    'Finish only after every required step has a reference. ' + instruction)
                if reference['action'] == 'finish_inputs':
                    draft['inputs_finished'] = True
                    save_author()
                    break
                state.remember_node_purpose(reference['node_ref'], reference['purpose'])
                references.append(reference['node_ref'])
                save_author()
            schema['properties']['input_refs'] = {'const': references}
    return dict(ask(router, 'construction_operation_' + op,
        {**current, **state.payload(choice_actions(schema, default_action='write_' + op),
            **{key: payload['construction_state'][key] for key in ('current_member', 'graph', 'sources')
               if key in payload['construction_state']})}, schema, instruction))
