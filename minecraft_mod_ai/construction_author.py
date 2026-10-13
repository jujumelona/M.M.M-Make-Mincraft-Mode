"""One bounded typed node per inference; host graph/inventory never enter prompts."""
from __future__ import annotations
import json
from copy import deepcopy
from itertools import islice
from .java_operation_ir import ATTRIBUTES, REQUIRED, OPERATIONS, JavaOperations
from .fixed_template_generation import _validate_native_arguments
from .construction_state import (ConstructionState, STATE_INSTRUCTION, PAGE_SIZE,
                                 page_result, symbol_page, symbol_record)

TEXT = {'type': 'string', 'maxLength': 240}
ID = {'type': 'string', 'maxLength': 240, 'minLength': 1}

def obj(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required, 'additionalProperties': False}

def array(items, maximum=8):
    return {'type': 'array', 'maxItems': maximum, 'items': items}

def ask(router, name, payload, schema, instruction):
    messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    return _validate_native_arguments(router.generate_structured_data('coder', messages, data_name=name, parameters=schema), schema)

NEXT_SCHEMA = obj({'action': {'type': 'string', 'enum': [*OPERATIONS, 'symbols', 'declarations',
                   'nodes', 'node_index', 'source', 'sources', 'complete', 'extension']},
                   'query': TEXT, 'offset': {'type': 'integer', 'minimum': 0}})


def operation_schema(op):
    scalar = {'anyOf': [{'type': 'string', 'maxLength': 240}, {'type': 'number'}, {'type': 'boolean'}, {'type': 'null'}]}
    attributes = {'value': scalar, 'name': ID, 'symbol_ref': ID, 'type_ref': ID, 'element_type': ID,
        'operator': {'type': 'string', 'maxLength': 4}, 'final': {'type': 'boolean'}, 'sized': {'type': 'boolean'},
        'finally': {'type': 'boolean'}, 'parameters': array(ID), 'cases': array(scalar),
        'catches': array(obj({'name': ID, 'type_ref': ID}))}
    return obj({'node_id': ID, 'input_refs': array(ID), 'typed_result': ID,
                'attributes': obj({key: attributes[key] for key in sorted(ATTRIBUTES[op])}, sorted(REQUIRED[op]))})


def _node_summary(row):
    return {k: row[k] for k in ('node_id', 'op', 'typed_result', 'input_refs')} | {
        'names': {k: v for k, v in row['attributes'].items() if k in {'name', 'parameters'}}}


def author_task_operations(router, task, context):
    symbols = context['symbols']
    declaration = symbols.get(task.owner_symbol)
    if declaration is None: raise ValueError('CONSTRUCTION_SYMBOL_MISSING: ' + task.owner_symbol)
    base = {'task_id': task.task_id, 'intent': task.intent, 'trigger': task.trigger, 'effects': task.effects,
            'state_reads': list(task.state_reads), 'state_writes': list(task.state_writes),
            'declaration': symbol_record(task.owner_symbol, declaration), 'target_path': task.target_path,
            'authoring_contract': context.get('authoring_contract', {}),
            'owned_runtime_records': context.get('owned_runtime_records', ()),
            'donor_call_obligations': context.get('donor_call_obligations', {}),
            'port_source_ref': context.get('port_source_ref', ''),
            'already_implemented_effects': context.get('already_implemented_effects', {})}
    resume = context.get('resume_nodes')
    existing, stage = resume(task) if callable(resume) else ((), '')
    nodes = [deepcopy(node) for node in existing]
    by_id = {row['node_id']: row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('CONSTRUCTION_CHECKPOINT_NODES_INVALID')
    if stage == 'validated':
        JavaOperations(task, nodes, symbols)
        return tuple(nodes)
    declaration_ids = context.get('construction_declaration_ids')
    if declaration_ids is None:
        declaration_ids = sorted(key for key, row in symbols.items()
                                 if row.get('owner') == declaration['owner'] and row.get('kind') != 'CLASS')
    state = ConstructionState(symbols, declaration['owner'], task.target_path,
        declaration_ids=declaration_ids, revision=len(nodes),
        authoring_contract=context.get('authoring_contract'), binding_pages=context.get('binding_pages'))
    state.enter('operation_graph')
    referenced = {ref for row in nodes for ref in row['input_refs']}
    pending = dict.fromkeys(ref for row in nodes for ref in row['input_refs'] if ref not in by_id)
    roots = dict.fromkeys(key for key in by_id if key not in referenced)
    source_units = context.get('source_pages', {})
    source_ids = sorted(source_units)
    bindings = context.get('binding_pages', {})
    read_refs = context.get('recorded_source_refs')
    read_sources = set(read_refs() if callable(read_refs) else ())

    def lookup_result(action, query, offset):
        if action == 'symbols': return symbol_page(symbols, query, offset)
        if action == 'declarations':
            return page_result([symbol_record(key, symbols[key]) for key in declaration_ids],
                               offset, field='declarations')
        if action == 'node_index':
            matches = [row for row in nodes if query.strip() in row['node_id']]
            result = page_result(matches, offset, field='nodes')
            result['nodes'] = [_node_summary(row) for row in result['nodes']]
            return result
        if action == 'nodes':
            keys = [key.strip() for key in query.split(',')]
            if offset != 0: raise ValueError('CONSTRUCTION_LOOKUP_OFFSET_INVALID')
            if len(keys) > PAGE_SIZE or any(key not in by_id for key in keys):
                raise ValueError('CONSTRUCTION_NODE_REFERENCE_MISSING')
            return {'nodes': [by_id[key] for key in keys], 'total_count': len(keys),
                    'status': 'matches', 'next_offset': None}
        if action == 'sources':
            matches = [{'source_ref': key, 'unit_count': len(source_units[key])} for key in source_ids
                       if query.strip().casefold() in key.casefold()]
            return page_result(matches, offset, field='sources')
        if action == 'bindings':
            return state.binding_result(query, offset)
        if action != 'source' or query not in source_units:
            raise ValueError('CONSTRUCTION_SOURCE_PAGE_MISSING')
        pages = source_units[query]
        if offset >= len(pages): raise ValueError('CONSTRUCTION_SOURCE_PAGE_OFFSET_INVALID')
        return {'source': pages[offset:offset + 1], 'total_count': len(pages),
                'status': 'matches', 'next_offset': offset + 1 if len(pages) > offset + 1 else None}

    resume_author = context.get('resume_author_state')
    saved_state = resume_author(task, tuple(nodes)) if callable(resume_author) else None
    if saved_state is not None:
        last = state.restore(saved_state)
        if last is not None:
            state.lookup = {**lookup_result(last['kind'], last['query'], last['offset']), **last}

    def save_author():
        callback = context.get('checkpoint_author_state')
        if callable(callback): callback(task, tuple(nodes), state.export())

    instruction = ('Author this current behavior as typed Java operations, one node per turn. No Java source strings. '
        'Use exact inspected symbol IDs and type names. Request symbols with a lexical query and offset, '
        'nodes by ID, or source by ID. Relevant graph nodes are stored on the host; do not recreate them. '
        'Implement only owned runtime records; never duplicate effects already provided by Typed/UI. '
        'When a pinned donor source is installed, use its exact inspected callable symbols '
        'through typed call/field/new operations and implement missing state and event glue. '
        'Only the latest four node summaries are shown. Use node_index to discover older node IDs, '
        'then nodes to read their complete records. Use sources to discover source IDs, then source '
        'to read one complete unit. Use bindings with a binding_id from authoring_contract.binding_pages '
        'to read complete configuration records. Respect the canonical responsibility, exact identities '
        'and implementation_requirements in authoring_contract. Forward references are allowed; pending_input_refs must be authored '
        'before complete. Finish only after all effects and return paths are represented. '
        'Unsupported syntax requires extension, never raw Java. ' + STATE_INSTRUCTION)
    while True:
        recent = [_node_summary(row) for row in nodes[-4:]]
        actions = list(NEXT_SCHEMA['properties']['action']['enum'])
        if bindings: actions.append('bindings')
        if not nodes or pending: actions.remove('complete')
        schema = obj({**NEXT_SCHEMA['properties'], 'action': {'type': 'string', 'enum': actions}})
        payload = {**base, 'recent_nodes': recent, **state.payload(actions,
            current_member=task.owner_symbol,
            graph={'accepted_node_count': len(nodes), 'resumed_node_count': len(existing),
                   'checkpoint_stage': stage or 'new',
                   'pending_input_ref_count': len(pending), 'pending_input_refs': list(islice(pending, PAGE_SIZE)),
                   'root_count': len(roots), 'root_ids': list(islice(roots, PAGE_SIZE))},
            sources={'total_count': len(source_ids), 'recent_ids': source_ids[:PAGE_SIZE],
                     'lookup_action': 'sources',
                     'pinned_source_read': bool(base['port_source_ref'] in read_sources)})}
        choice = ask(router, 'construction_operation_next', payload, schema, instruction)
        action = choice['action']
        if action == 'extension': raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: ' + choice['query'])
        if action == 'complete':
            JavaOperations(task, nodes, symbols)
            if context.get('checkpoint'): context['checkpoint'](task, tuple(nodes), 'validated')
            return tuple(nodes)
        if action in {'symbols', 'declarations', 'nodes', 'node_index', 'source', 'sources', 'bindings'}:
            query = choice['query'].strip()
            if action == 'declarations': query = ''
            state.consume_lookup(action, query, choice['offset'],
                                 lookup_result(action, query, choice['offset']))
            if action == 'source':
                callback = context.get('source_read')
                if callable(callback):
                    callback(task, query)
                read_sources.add(query)
            save_author()
            continue
        row = dict(ask(router, 'construction_operation_' + action,
                       {**payload, 'selected_operation': action}, operation_schema(action), instruction))
        row['op'] = action
        if row['node_id'] in by_id: raise ValueError('CONSTRUCTION_NODE_DUPLICATE')
        by_id[row['node_id']] = row; nodes.append(row)
        pending.pop(row['node_id'], None)
        for ref in row['input_refs']:
            if ref not in by_id: pending.setdefault(ref, None)
        if row['node_id'] not in referenced: roots[row['node_id']] = None
        for ref in row['input_refs']: roots.pop(ref, None)
        referenced.update(row['input_refs'])
        state.advance()
        stage = 'authoring'
        if context.get('checkpoint'): context['checkpoint'](task, tuple(nodes), 'authoring')
        save_author()
