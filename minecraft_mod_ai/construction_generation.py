"""Stream class/member contracts, then render their independently authored graphs."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from .construction_author import obj, ask
from .construction_state import ConstructionState, STATE_INSTRUCTION
from .construction_declaration_choices import (header_candidates, header_schema, choice_actions,
    member_schema, parameter_schema, consume_lookup_choice, choose_type, with_binding_lookups)
from .construction_task import ConstructionTask
from .construction_author import author_task_operations
from .construction_java import render_task_member, render_task_java
from .java_operation_ir import PRIMITIVES, identifier

SIGNATURE = obj({'visibility': {'type': 'string', 'enum': ['public', 'protected', 'private', '']},
                 'static': {'type': 'boolean'}, 'final': {'type': 'boolean'}})
HEADER_INSTRUCTION = ('Choose only inheritance for the current new class. No Java source. '
    'Complete this phase when no further superclass or interfaces are needed; methods and behavior bodies '
    'are planned and authored in later phases. A class with no explicit superclass extends Object implicitly. '
    'Search dependencies with symbols; inheritance can use only the inspected header_candidates in this page. '
    'A superclass is a non-final class. An implemented type is an interface. The new owner is neither '
    'an inheritance candidate nor an implementation target in this phase. ' + STATE_INSTRUCTION)
MEMBER_INSTRUCTION = ('Plan only additional helper methods, fields or constructors. No Java source. '
    'Declarations with a body_owner already have an assigned author and will receive their actual bodies '
    'after this phase. Do not redeclare them. Complete means no additional helper declarations are needed; '
    'begin_bodies selects the first queued body to author. Both transitions keep every assigned body. '
    'Use overload only to add a distinct signature to a helper. Use symbols for dependency signatures and '
    'declarations for older members. Types are exact inspected names; primitives are intrinsic. ' + STATE_INSTRUCTION)
SIGNATURE_INSTRUCTION = ('Describe only the current new member signature: return type, visibility and '
    'modifiers. No Java source. Use exact inspected type names or primitives. This phase does not declare '
    'other members or finish their behaviors. ' + STATE_INSTRUCTION)
PARAMETER_INSTRUCTION = ('Plan exactly one parameter for the current method or constructor. '
    'Complete means its parameter list is finished, not that its body is finished. A parameter has a Java '
    'identifier and an exact inspected type or primitive. Do not emit method/class declarations. ' + STATE_INSTRUCTION)


def exact_type(value, symbols, *, allow_void=True):
    base = value
    while base.endswith('[]'):
        base = base[:-2]
    if base in PRIMITIVES:
        if base == 'void' and (value != base or not allow_void):
            raise ValueError('CONSTRUCTION_TYPE_VOID')
    elif symbols.get(base, {}).get('kind') != 'CLASS':
        raise ValueError('CONSTRUCTION_TYPE_NOT_INSPECTED: ' + base)
    return value


def compose(router, spec, *, source_context=(), construction_context=None,
            authoring_contract=None, binding_pages=None):
    from .first_pass_java_source import _java_identity
    identity = _java_identity(spec)
    if identity is None:
        raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: Java fragment/identity')
    if not construction_context or 'symbols' not in construction_context:
        raise ValueError('CONSTRUCTION_TARGET_CONTEXT_REQUIRED')
    context = dict(construction_context)
    context['authoring_contract'] = dict(authoring_contract or {})
    context['binding_pages'] = dict(binding_pages or {})
    symbols = dict(context['symbols'])
    context['symbols'] = symbols
    package, name = identity
    owner = package + '.' + name
    bindings = spec.get('bindings', {})
    if not isinstance(bindings, dict):
        raise ValueError('CONSTRUCTION_BINDINGS_REQUIRED')
    from .construction_journal import JournaledConstructionRouter
    planning_router = JournaledConstructionRouter(
        router, context['project_root'], spec, context)
    saved = bindings.get('construction_task_json')
    task = ConstructionTask.from_dict(json.loads(saved)) if saved else ConstructionTask(
        'canonical_' + hashlib.sha256(spec['target_path'].encode()).hexdigest()[:16],
        (str(bindings.get('requirement_id') or spec.get('context_id') or 'canonical'),), 'logic', spec['requirement'],
        target_path=spec['target_path'], allowed_writes=(spec['target_path'],), effects=spec['requirement'])
    behaviors = json.loads(bindings.get('construction_behaviors_json') or '[]')
    if bindings.get('request_feature_entrypoint'):
        if not isinstance(behaviors, list) or not behaviors:
            raise ValueError('CONSTRUCTION_BEHAVIOR_COVERAGE_MISSING')
        expected = str(bindings.get('parent_requirement_id') or bindings.get('requirement_id') or '')
        ids = [row.get('behavior_id') for row in behaviors]
        if len(ids) != len(set(ids)) or any(
            not isinstance(row, dict) or not row.get('intent') or not row.get('effects')
            or not row.get('source_quote') or (expected and row.get('requirement_id') != expected)
            for row in behaviors
        ):
            raise ValueError('CONSTRUCTION_BEHAVIOR_SCOPE_INVALID')
    if task.target_path and task.target_path != spec['target_path']:
        raise ValueError('CONSTRUCTION_TARGET_MISMATCH')
    task = replace(task, target_path=spec['target_path'], allowed_writes=(spec['target_path'],))
    if bindings.get('request_feature_entrypoint'):
        lifecycle, interface = 'onInitialize', 'net.fabricmc.api.ModInitializer'
    elif str(spec.get('side', '')).upper() == 'CLIENT':
        lifecycle, interface = 'onInitializeClient', 'net.fabricmc.api.ClientModInitializer'
    else:
        lifecycle = interface = ''
    # This renderer owns a new complete class, not edits to an inspected class.
    # Never relabel installed source or bytecode as a provisional declaration.
    if owner in symbols or any(row.get('owner') == owner for row in symbols.values()):
        raise ValueError('CONSTRUCTION_OWNER_ALREADY_EXISTS: ' + owner)
    interfaces, superclass = [], ''
    symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': name, 'access': 1,
        'superclass': superclass, 'interfaces': interfaces,
        'side': str(spec.get('side', 'COMMON')), 'provenance': 'construction_plan'}
    state = ConstructionState(symbols, owner, task.target_path,
        authoring_contract=context['authoring_contract'], binding_pages=context['binding_pages'])
    payload = {'intent': task.intent, 'trigger': task.trigger, 'effects': task.effects,
               'state_reads': task.state_reads, 'state_writes': task.state_writes,
               'owner': owner, 'target_path': task.target_path,
               'authoring_contract': context['authoring_contract']}
    if interface:
        exact_type(interface, symbols, allow_void=False)
        interfaces.append(interface)
        payload['host_entrypoint'] = {'interface': interface, 'method': lifecycle}
        state.advance()
    else:
        state.enter('class_header')
        while True:
            candidates = header_candidates(state)
            schema = header_schema(state, candidates)
            actions = choice_actions(schema)
            choice = ask(planning_router, 'construction_class_next',
                {**payload, **state.payload(actions, header_candidates=candidates)}, schema, HEADER_INSTRUCTION)
            if choice['action'] == 'complete':
                break
            if choice['action'] == 'extension':
                raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: ' + choice['reason'])
            if consume_lookup_choice(state, choice):
                continue
            type_ = exact_type(choice['type_ref'], symbols, allow_void=False)
            if type_ not in candidates[choice['action']]:
                raise ValueError('CONSTRUCTION_CLASS_HEADER')
            if choice['action'] == 'superclass':
                if superclass:
                    raise ValueError('CONSTRUCTION_CLASS_HEADER')
                superclass = type_
                symbols[owner]['superclass'] = superclass
            else:
                if type_ in interfaces:
                    raise ValueError('CONSTRUCTION_CLASS_HEADER')
                interfaces.append(type_)
            state.advance()
    declarations = []
    registrars = []
    if bindings.get('request_feature_entrypoint'):
        # Each preserved child behavior owns a distinct method. Host lifecycle
        # code registers every one, while MiMo writes the actual callbacks.
        for behavior in behaviors:
            symbol_name = 'registerBehavior' + hashlib.sha256(
                behavior['behavior_id'].encode('utf-8')).hexdigest()[:20]
            key = owner + '#' + symbol_name + '()'
            if key in symbols:
                raise ValueError('CONSTRUCTION_DECLARATION_DUPLICATE: ' + key)
            symbols[key] = {'kind': 'METHOD', 'owner': owner, 'name': symbol_name,
                'parameters': [], 'return_type': 'void', 'static': False,
                'visibility': 'private', 'side': str(spec.get('side', 'COMMON')),
                'class_superclass': superclass, 'class_interfaces': interfaces,
                'provenance': 'construction_plan', 'body_owner': behavior['behavior_id'],
                'purpose': behavior['intent']}
            registrars.append((key, behavior))
            state.add_declaration(key)
    if lifecycle:
        key = owner + '#' + lifecycle + '()'
        if key in symbols:
            raise ValueError('CONSTRUCTION_HOST_ENTRYPOINT_DUPLICATE')
        symbols[key] = {'kind': 'METHOD', 'owner': owner, 'name': lifecycle, 'parameters': [],
            'return_type': 'void', 'static': False, 'visibility': 'public',
            'annotations': ['java.lang.Override'], 'class_superclass': superclass,
            'class_interfaces': interfaces, 'side': str(spec.get('side', 'COMMON')),
            'provenance': 'construction_plan',
            'body_owner': 'host_lifecycle' if registrars else task.task_id,
            'purpose': 'Invoke every assigned behavior registrar' if registrars else task.intent}
        state.add_declaration(key)
    reserved_names = {symbols[key]['name'] for key, _behavior in registrars}
    if lifecycle:
        reserved_names.add(lifecycle)
    preferred_body = ''
    while True:
        state.enter('member_declarations')
        # The page is already bounded; every remaining body still stays queued.
        body_candidates = [key for key in state.declaration_ids if
                           symbols[key].get('body_owner') != 'host_lifecycle'][-8:]
        schema = member_schema(state, reserved_names=reserved_names, body_candidates=body_candidates)
        actions = choice_actions(schema)
        item = ask(planning_router, 'construction_member_next',
            {**payload, **state.payload(actions, body_candidates=body_candidates)}, schema, MEMBER_INSTRUCTION)
        action = item['action']
        if action == 'complete':
            break
        if action == 'begin_bodies':
            preferred_body = item['symbol_ref']
            break
        if action == 'extension':
            raise ValueError('CONSTRUCTION_RENDERER_EXTENSION_REQUIRED: ' + item['reason'])
        if consume_lookup_choice(state, item):
            continue
        if action == 'declarations':
            state.lookup_declarations(item['offset'])
            continue
        member_name = name if action == 'constructor' else identifier(item['name'])
        if member_name in reserved_names:
            raise ValueError('CONSTRUCTION_HOST_DECLARATION_RESERVED')
        if action == 'overload':
            action = 'method'
        current = {**payload, 'kind': action, 'name': member_name, 'purpose': item['purpose']}
        return_type = 'void' if action == 'constructor' else choose_type(planning_router, state, current,
            phase='member_return_type', allow_void=action != 'field')
        state.enter('member_signature')
        schema = obj({**SIGNATURE['properties'], 'return_type': {'const': return_type}})
        if action == 'constructor':
            schema['properties']['static'] = {'const': False}
        while True:
            decision_schema = with_binding_lookups(state, schema)
            metadata = ask(planning_router, 'construction_member_signature',
                {**current, 'return_type': return_type,
                 **state.payload(choice_actions(decision_schema, default_action='signature'))},
                decision_schema, SIGNATURE_INSTRUCTION)
            if consume_lookup_choice(state, metadata):
                continue
            break
        exact_type(metadata['return_type'], symbols, allow_void=action != 'field')
        state.advance()
        parameters = []
        if action != 'field':
            state.enter('member_parameters')
            while True:
                # Same-name helpers can be overloaded only by a distinct
                # parameter signature. The sampler cannot finish a duplicate.
                candidate_key = owner + '#' + member_name + '(' + ','.join(p['type'] for p in parameters) + ')'
                schema = parameter_schema(parameters, state=state, can_complete=candidate_key not in symbols)
                parameter = ask(planning_router, 'construction_parameter_next',
                    {**current, 'signature': metadata, 'parameters': parameters,
                     'parameter_index': len(parameters),
                     **state.payload(choice_actions(schema))},
                    schema, PARAMETER_INSTRUCTION)
                if consume_lookup_choice(state, parameter):
                    continue
                if parameter['complete']:
                    break
                identifier(parameter['name'])
                type_ = choose_type(planning_router, state,
                    {**current, 'signature': metadata, 'parameters': parameters, 'parameter_name': parameter['name']},
                    phase='member_parameter_type')
                exact_type(type_, symbols, allow_void=False)
                if any(p['name'] == parameter['name'] for p in parameters):
                    raise ValueError('CONSTRUCTION_PARAMETER_DUPLICATE')
                parameters.append({'name': parameter['name'], 'type': type_})
                state.advance()
                state.enter('member_parameters')
        key = owner + '#' + member_name + '(' + ','.join(p['type'] for p in parameters) + ')'
        if key in symbols:
            raise ValueError('CONSTRUCTION_DECLARATION_DUPLICATE: ' + key)
        symbols[key] = {**metadata, 'kind': action.upper(), 'owner': owner, 'name': member_name,
            'parameters': parameters, 'class_superclass': superclass, 'class_interfaces': interfaces,
            'side': str(spec.get('side', 'COMMON')), 'provenance': 'construction_plan',
            'body_owner': task.task_id, 'purpose': item['purpose']}
        declarations.append((key, item['purpose']))
        state.add_declaration(key)
    if lifecycle and not registrars:
        declarations.append((owner + '#' + lifecycle + '()', task.intent))
    if not declarations and not registrars:
        raise ValueError('CONSTRUCTION_BODY_EMPTY')
    assigned_records = {}
    residual_records = json.loads(bindings.get('owned_runtime_records_json') or '[]')
    typed_owners = json.loads(bindings.get('retained_typed_behavior_json') or '{}')
    if registrars and residual_records:
        from .construction_ownership import assign_behavior_records
        assigned_records = assign_behavior_records(planning_router, behaviors, residual_records)

    donor_symbols = context.get('donor_callable_symbols', {})
    from .construction_porting import select_port_tasks
    port_tasks = select_port_tasks(
        planning_router, task, owner=owner,
        references=context.get('reference_port_obligations', ()),
        symbols=symbols, exact_type=exact_type, state=state,
    )
    context['construction_declaration_ids'] = tuple(state.declaration_ids)
    from .construction_progress import planned_declarations_fingerprint, construction_input_fingerprint
    planned_declaration_hash = planned_declarations_fingerprint(symbols, state.declaration_ids)
    input_hash = construction_input_fingerprint(spec, context['authoring_contract'])
    member_invocations = {}

    def author_one(member_task, *, owned_records=(), port_source_ref=''):
        from .construction_progress import ConstructionNodeCheckpoint
        if not context.get('project_root'):
            raise ValueError('CONSTRUCTION_RESUME_WORKSPACE_REQUIRED')
        progress = ConstructionNodeCheckpoint(context['project_root'], member_task,
            target=context['target'],
            prerequisite_hashes=context['prerequisite_hashes'],
            declaration=symbols[member_task.owner_symbol], owner_header=symbols[owner],
            planned_declaration_hash=planned_declaration_hash, construction_input_hash=input_hash)
        scoped_context = dict(context)
        scoped_context['owned_runtime_records'] = list(owned_records)
        scoped_context['port_source_ref'] = port_source_ref
        scoped_context['source_read'] = progress.record_source_read
        scoped_context['donor_call_obligations'] = {
            cap: {'callable_symbol_count': len(identifiers),
                  'lookup_hint': identifiers[0].split('#', 1)[0]}
            for cap, identifiers in donor_symbols.items()
        }
        scoped_context['already_implemented_effects'] = {
            'ui_owned_records': typed_owners.get('ui_owned_records', ()),
            'typed_functions': typed_owners.get('functions', ()),
            'typed_events': typed_owners.get('event_bindings', ()),
        }
        scoped_context['resume_nodes'] = lambda current: progress.load()
        scoped_context['checkpoint'] = progress.save
        scoped_context['resume_author_state'] = progress.load_author_state
        scoped_context['checkpoint_author_state'] = progress.save_author_state
        scoped_context['recorded_source_refs'] = progress.recorded_source_refs
        nodes = author_task_operations(router, member_task, scoped_context)
        if port_source_ref and port_source_ref not in progress.recorded_source_refs():
            raise ValueError('CONSTRUCTION_PORT_SOURCE_NOT_READ: ' + port_source_ref)
        member_invocations[member_task.owner_symbol] = {
            node['attributes']['symbol_ref']
            for node in nodes
            if node['op'] in {'call', 'new', 'field'}
            and node['attributes'].get('symbol_ref')
        }
        return nodes

    body_jobs = []
    for index, (key, purpose) in enumerate(declarations):
        member_task = replace(task, task_id=task.task_id + '.member.' + str(index), owner_symbol=key,
                              intent=purpose, output_symbols=(key,))
        body_jobs.append((member_task, {}))
    for port_task, reference_id in port_tasks:
        body_jobs.append((port_task, {'port_source_ref': reference_id}))
    for key, behavior in registrars:
        registrar_task = replace(task,
            task_id=behavior['behavior_id'] + '.registration',
            requirement_ids=(behavior['requirement_id'],),
            behavior_id=behavior['behavior_id'], owner_symbol=key,
            intent=behavior['intent'], trigger=behavior.get('trigger', ''),
            effects=behavior['effects'],
            state_reads=tuple(behavior.get('state_reads', ())),
            state_writes=tuple(behavior.get('state_writes', ())),
            output_symbols=(key,),
            verification_ids=(behavior['behavior_id'] + '.compile',
                              behavior['behavior_id'] + '.behavior'))
        owned = [record for record in residual_records if
                 assigned_records.get(record['record_ref']) == behavior['behavior_id']]
        body_jobs.append((registrar_task, {'owned_records': owned}))
    # A request to author an already-assigned method changes the phase, never
    # recreates its signature or removes other pending behavior owners.
    authoring_order = sorted(body_jobs, key=lambda job: job[0].owner_symbol != preferred_body)
    rendered = {}
    for member_task, options in authoring_order:
        nodes = author_one(member_task, **options)
        render_task_java(member_task, nodes, symbols)
        rendered[member_task.owner_symbol] = render_task_member(member_task, nodes, symbols)
    # Keep declaration/field initialization order independent of author order.
    members = [rendered[current.owner_symbol] for current, _options in body_jobs]
    # Donor/port calls in an unreachable helper are not runtime integration.
    # The lifecycle invokes every registrar; follow its typed member call
    # graph transitively before claiming source glue was actually connected.
    roots = [key for key, _behavior in registrars]
    if not roots:
        roots = list(member_invocations)
    reachable_members, reachable_calls, frontier = set(), set(), list(roots)
    while frontier:
        current = frontier.pop()
        if current in reachable_members:
            continue
        reachable_members.add(current)
        for called in member_invocations.get(current, ()):
            reachable_calls.add(called)
            if called in member_invocations and called not in reachable_members:
                frontier.append(called)
    missing_port_calls = [
        item.owner_symbol for item, _ref in port_tasks
        if item.owner_symbol not in reachable_calls
    ]
    if missing_port_calls:
        raise ValueError('CONSTRUCTION_PORT_GLUE_CALL_MISSING: ' +
                         ','.join(sorted(missing_port_calls)))
    missing_donor_calls = [
        cap for cap, identifiers in donor_symbols.items()
        if not set(identifiers) & reachable_calls
    ]
    if missing_donor_calls:
        raise ValueError('CONSTRUCTION_DONOR_GLUE_CALL_MISSING: ' +
                         ','.join(sorted(missing_donor_calls)))
    if registrars:
        lifecycle_calls = '\n'.join('this.' + symbols[key]['name'] + '();'
                                    for key, _ in registrars)
        members.append('@Override\npublic void onInitialize() {\n' + lifecycle_calls + '\n}')
    source = 'package ' + package + ';\n\npublic ' + ('final ' if lifecycle else '') + 'class ' + name
    source += (' extends ' + superclass if superclass else '') + (' implements ' + ', '.join(interfaces) if interfaces else '')
    source += ' {\n' + '\n\n'.join(members) + '\n}\n'
    if context.get('verify_source'):
        context['verify_source'](source, spec['target_path'])
    from .construction_coverage import record_authored_source_calls
    record_authored_source_calls(context['project_root'], spec['target_path'], source,
        selected_ports=[
            {'task_id': port_task.task_id, 'source_ref': ref,
             'owner_symbol': port_task.owner_symbol}
            for port_task, ref in port_tasks
        ],
        installed_donor_calls=sorted(
            reachable_calls & {
                ref for rows in donor_symbols.values() for ref in rows
            }
        ),
    )
    return source
