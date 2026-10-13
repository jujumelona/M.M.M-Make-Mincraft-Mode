"""Stream class/member contracts, then render their independently authored graphs."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from .construction_author import ID, TEXT, obj, ask, symbol_page
from .construction_task import ConstructionTask
from .construction_author import author_task_operations
from .construction_java import render_task_member, render_task_java
from .java_operation_ir import PRIMITIVES, identifier, qualified_name

CLASS_NEXT = obj({'action': {'type': 'string', 'enum': ['superclass', 'interface', 'symbols', 'complete']},
                  'type_ref': TEXT, 'offset': {'type': 'integer', 'minimum': 0}})
MEMBER_NEXT = obj({'action': {'type': 'string', 'enum': ['method', 'field', 'constructor', 'symbols', 'complete', 'extension']},
                  'name': TEXT, 'purpose': TEXT, 'query': TEXT, 'offset': {'type': 'integer', 'minimum': 0}})
SIGNATURE = obj({'return_type': ID, 'visibility': {'type': 'string', 'enum': ['public', 'protected', 'private', '']},
                 'static': {'type': 'boolean'}, 'final': {'type': 'boolean'}})
PARAMETER = obj({'complete': {'type': 'boolean'}, 'name': TEXT, 'type_ref': TEXT})
INSTRUCTION = ('Plan one current Java declaration at a time. No source code. Types are exact fully qualified '
    'inspected inventory names (primitives are intrinsic). Request symbols using a lexical query and offset. '
    'All existing signatures are retrievable by lookup. Do not duplicate a declaration. Complete only after '
    'every required behavior has an owner. Use extension for unsupported language constructs.')


def exact_type(value, symbols, *, allow_void=True):
    base = value
    while base.endswith('[]'): base = base[:-2]
    if base in PRIMITIVES:
        if base == 'void' and (value != base or not allow_void): raise ValueError('CONSTRUCTION_TYPE_VOID')
    elif symbols.get(base, {}).get('kind') != 'CLASS':
        raise ValueError('CONSTRUCTION_TYPE_NOT_INSPECTED: ' + base)
    return value


def compose(router, spec, *, source_context=(), construction_context=None):
    from .first_pass_java_source import _java_identity
    identity = _java_identity(spec)
    if identity is None: raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: Java fragment/identity')
    if not construction_context or 'symbols' not in construction_context:
        raise ValueError('CONSTRUCTION_TARGET_CONTEXT_REQUIRED')
    context = dict(construction_context)
    symbols = dict(context['symbols']); context['symbols'] = symbols
    package, name = identity; owner = package + '.' + name
    bindings = spec.get('bindings', {})
    if not isinstance(bindings, dict): raise ValueError('CONSTRUCTION_BINDINGS_REQUIRED')
    saved = bindings.get('construction_task_json')
    task = ConstructionTask.from_dict(json.loads(saved)) if saved else ConstructionTask(
        'canonical_' + hashlib.sha256(spec['target_path'].encode()).hexdigest()[:16],
        (str(bindings.get('requirement_id') or spec.get('context_id') or 'canonical'),), 'logic', spec['requirement'],
        target_path=spec['target_path'], allowed_writes=(spec['target_path'],), effects=spec['requirement'])
    if task.target_path and task.target_path != spec['target_path']: raise ValueError('CONSTRUCTION_TARGET_MISMATCH')
    task = replace(task, target_path=spec['target_path'], allowed_writes=(spec['target_path'],))
    if bindings.get('request_feature_entrypoint'):
        lifecycle, interface = 'onInitialize', 'net.fabricmc.api.ModInitializer'
    elif str(spec.get('side', '')).upper() == 'CLIENT':
        lifecycle, interface = 'onInitializeClient', 'net.fabricmc.api.ClientModInitializer'
    else: lifecycle = interface = ''
    interfaces, superclass, page, seen = [], '', {}, set()
    payload = {'intent': task.intent, 'trigger': task.trigger, 'effects': task.effects,
               'state_reads': task.state_reads, 'state_writes': task.state_writes,
               'owner': owner, 'target_path': task.target_path}
    if interface:
        exact_type(interface, symbols, allow_void=False); interfaces.append(interface)
        payload['host_entrypoint'] = {'interface': interface, 'method': lifecycle}
    else:
        while True:
            choice = ask(router, 'construction_class_next', {**payload, 'lookup': page}, CLASS_NEXT, INSTRUCTION)
            if choice['action'] == 'complete': break
            fingerprint = json.dumps(choice, sort_keys=True)
            if fingerprint in seen: raise ValueError('CONSTRUCTION_AUTHOR_NO_PROGRESS')
            seen.add(fingerprint)
            if choice['action'] == 'symbols':
                page = symbol_page(symbols, choice['type_ref'], choice['offset']); continue
            type_ = exact_type(choice['type_ref'], symbols, allow_void=False)
            if choice['action'] == 'superclass':
                if superclass: raise ValueError('CONSTRUCTION_CLASS_HEADER')
                superclass = type_
            else: interfaces.append(type_)
            page = {}
    symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'access': 1, 'superclass': superclass,
                      'interfaces': interfaces, 'side': str(spec.get('side', 'COMMON'))}
    declarations, seen, page = [], set(), {}
    while True:
        item = ask(router, 'construction_member_next', {**payload, 'lookup': page}, MEMBER_NEXT, INSTRUCTION)
        action = item['action']
        if action == 'complete': break
        if action == 'extension': raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: ' + item['purpose'])
        if action == 'symbols':
            fingerprint = ('lookup', item['query'], item['offset'])
            if fingerprint in seen: raise ValueError('CONSTRUCTION_AUTHOR_NO_PROGRESS')
            seen.add(fingerprint); page = symbol_page(symbols, item['query'], item['offset']); continue
        member_name = name if action == 'constructor' else identifier(item['name'])
        if member_name == lifecycle: raise ValueError('CONSTRUCTION_HOST_ENTRYPOINT_DUPLICATE')
        current = {**payload, 'kind': action, 'name': member_name, 'purpose': item['purpose'], 'lookup': page}
        metadata = ask(router, 'construction_member_signature', current, SIGNATURE, INSTRUCTION)
        exact_type(metadata['return_type'], symbols, allow_void=action != 'field')
        parameters = []
        if action != 'field':
            while True:
                parameter = ask(router, 'construction_parameter_next', {**current, 'parameter_index': len(parameters)}, PARAMETER, INSTRUCTION)
                if parameter['complete']: break
                identifier(parameter['name']); exact_type(parameter['type_ref'], symbols, allow_void=False)
                if any(p['name'] == parameter['name'] for p in parameters): raise ValueError('CONSTRUCTION_PARAMETER_DUPLICATE')
                parameters.append({'name': parameter['name'], 'type': parameter['type_ref']})
        key = owner + '#' + member_name + '(' + ','.join(p['type'] for p in parameters) + ')'
        if key in symbols: raise ValueError('CONSTRUCTION_DECLARATION_DUPLICATE: ' + key)
        symbols[key] = {**metadata, 'kind': action.upper(), 'owner': owner, 'name': member_name,
            'parameters': parameters, 'class_superclass': superclass, 'class_interfaces': interfaces,
            'side': str(spec.get('side', 'COMMON'))}
        declarations.append((key, item['purpose']))
        page = {}; seen.clear()
    if lifecycle:
        key = owner + '#' + lifecycle + '()'
        symbols[key] = {'kind': 'METHOD', 'owner': owner, 'name': lifecycle, 'parameters': [],
            'return_type': 'void', 'static': False, 'visibility': 'public', 'annotations': ['java.lang.Override'],
            'class_interfaces': interfaces, 'side': str(spec.get('side', 'COMMON'))}
        declarations.append((key, task.intent))
    if not declarations: raise ValueError('CONSTRUCTION_BODY_EMPTY')
    members = []
    for index, (key, purpose) in enumerate(declarations):
        member_task = replace(task, task_id=task.task_id + '.member.' + str(index), owner_symbol=key,
                              intent=purpose, output_symbols=(key,))
        nodes = author_task_operations(router, member_task, context)
        render_task_java(member_task, nodes, symbols)
        members.append(render_task_member(member_task, nodes, symbols))
    source = 'package ' + package + ';\n\npublic ' + ('final ' if lifecycle else '') + 'class ' + name
    source += (' extends ' + superclass if superclass else '') + (' implements ' + ', '.join(interfaces) if interfaces else '')
    source += ' {\n' + '\n\n'.join(members) + '\n}\n'
    if context.get('verify_source'): context['verify_source'](source, spec['target_path'])
    return source
