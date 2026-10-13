"""One bounded typed node per inference; host graph/inventory never enter prompts."""
from __future__ import annotations
import json
from copy import deepcopy
from .java_operation_ir import ATTRIBUTES, REQUIRED, OPERATIONS, JavaOperations
from .fixed_template_generation import _validate_native_arguments

TEXT = {'type': 'string', 'maxLength': 240}
ID = {'type': 'string', 'maxLength': 240, 'minLength': 1}

def obj(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required, 'additionalProperties': False}

def array(items, maximum=8):
    return {'type': 'array', 'maxItems': maximum, 'items': items}

def ask(router, name, payload, schema, instruction):
    messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    return _validate_native_arguments(router.generate_structured_data('coder', messages, data_name=name, parameters=schema), schema)

NEXT_SCHEMA = obj({'action': {'type': 'string', 'enum': [*OPERATIONS, 'symbols', 'nodes', 'source', 'complete', 'extension']},
                   'query': TEXT, 'offset': {'type': 'integer', 'minimum': 0}})


def operation_schema(op):
    scalar = {'anyOf': [{'type': 'string', 'maxLength': 240}, {'type': 'number'}, {'type': 'boolean'}, {'type': 'null'}]}
    attributes = {'value': scalar, 'name': ID, 'symbol_ref': ID, 'type_ref': ID, 'element_type': ID,
        'operator': {'type': 'string', 'maxLength': 4}, 'final': {'type': 'boolean'}, 'sized': {'type': 'boolean'},
        'finally': {'type': 'boolean'}, 'parameters': array(ID), 'cases': array(scalar),
        'catches': array(obj({'name': ID, 'type_ref': ID}))}
    return obj({'node_id': ID, 'input_refs': array(ID), 'typed_result': ID,
                'attributes': obj({key: attributes[key] for key in sorted(ATTRIBUTES[op])}, sorted(REQUIRED[op]))})


def symbol_page(symbols, query, offset):
    if len(query.strip()) < 2: raise ValueError('CONSTRUCTION_LOOKUP_QUERY_REQUIRED')
    matches = sorted(key for key, row in symbols.items() if query.casefold() in key.casefold() or query.casefold() == str(row.get('name', '')).casefold())
    return {'matches': [{'symbol_id': key, **{k: v for k, v in symbols[key].items() if k not in {'code', 'source', 'source_path'}}}
                        for key in matches[offset:offset + 8]], 'next_offset': offset + 8 if len(matches) > offset + 8 else None}


def author_task_operations(router, task, context):
    symbols = context['symbols']
    declaration = symbols.get(task.owner_symbol)
    if declaration is None: raise ValueError('CONSTRUCTION_SYMBOL_MISSING: ' + task.owner_symbol)
    base = {'task_id': task.task_id, 'intent': task.intent, 'trigger': task.trigger, 'effects': task.effects,
            'state_reads': list(task.state_reads), 'state_writes': list(task.state_writes),
            'declaration': declaration, 'target_path': task.target_path}
    resume = context.get('resume_nodes')
    existing, stage = resume(task) if callable(resume) else ((), '')
    nodes = [deepcopy(node) for node in existing]
    by_id = {row['node_id']: row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('CONSTRUCTION_CHECKPOINT_NODES_INVALID')
    if stage == 'validated':
        JavaOperations(task, nodes, symbols)
        return tuple(nodes)
    page, lookups = {}, set()
    instruction = ('Author this current behavior as typed Java operations, one node per turn. No Java source strings. '
        'Use exact inspected symbol IDs and type names. Request symbols with a lexical query and offset, '
        'nodes by ID, or source by ID. Relevant graph nodes are stored on the host; do not recreate them. '
        'Only the latest four node summaries are shown. Forward references are allowed. Finish only after '
        'all effects and return paths are represented. Unsupported syntax requires extension, never raw Java.')
    while True:
        recent = [{k: row[k] for k in ('node_id', 'op', 'typed_result')} for row in nodes[-4:]]
        payload = {**base, 'recent_nodes': recent, 'lookup': page}
        choice = ask(router, 'construction_operation_next', payload, NEXT_SCHEMA, instruction)
        action = choice['action']
        if action == 'extension': raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: ' + choice['query'])
        if action == 'complete':
            JavaOperations(task, nodes, symbols)
            if context.get('checkpoint'): context['checkpoint'](task, tuple(nodes), 'validated')
            return tuple(nodes)
        if action in {'symbols', 'nodes', 'source'}:
            request = (action, choice['query'], choice['offset'])
            if request in lookups: raise ValueError('CONSTRUCTION_AUTHOR_NO_PROGRESS: repeated lookup')
            lookups.add(request)
            if action == 'symbols': page = symbol_page(symbols, choice['query'], choice['offset'])
            elif action == 'nodes':
                keys = [key.strip() for key in choice['query'].split(',')]
                if len(keys) > 8 or any(key not in by_id for key in keys): raise ValueError('CONSTRUCTION_NODE_REFERENCE_MISSING')
                page = {'nodes': [by_id[key] for key in keys]}
            else:
                units = context.get('source_pages', {})
                if choice['query'] not in units: raise ValueError('CONSTRUCTION_SOURCE_PAGE_MISSING')
                pages = units[choice['query']]
                page = {'source': pages[choice['offset']:choice['offset'] + 1]}
            continue
        row = dict(ask(router, 'construction_operation_' + action, payload, operation_schema(action), instruction))
        row['op'] = action
        if row['node_id'] in by_id: raise ValueError('CONSTRUCTION_NODE_DUPLICATE')
        by_id[row['node_id']] = row; nodes.append(row)
        page = {}; lookups.clear()
        if context.get('checkpoint'): context['checkpoint'](task, tuple(nodes), 'authoring')
