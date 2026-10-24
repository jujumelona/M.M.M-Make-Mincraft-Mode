"""Stream class/member contracts, then render their independently authored graphs."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from .construction_author import obj, ask
from .construction_state import ConstructionState, STATE_INSTRUCTION, symbol_record
from .construction_declaration_choices import (header_candidates, header_schema, choice_actions,
    member_schema, parameter_schema, consume_lookup_choice, choose_type, with_binding_lookups)
from .construction_task import ConstructionTask
from .construction_author import author_task_operations
from .construction_java import render_task_member, render_task_java
from .java_operation_ir import PRIMITIVES, identifier, invocation_parameter_types

SIGNATURE = obj({'visibility': {'type': 'string', 'enum': ['public', 'protected', 'private', '']},
                 'static': {'type': 'boolean'}, 'final': {'type': 'boolean'}})
HEADER_INSTRUCTION = ('Choose only inheritance for the current new class. No Java source. '
    'finish_header ends only this phase when no further superclass or interfaces are needed; methods and behavior bodies '
    'are planned and authored in later phases. A class with no explicit superclass extends Object implicitly. '
    'Dependency API names in host_requirements are not required supertypes. Use inheritance only when '
    'the artifact responsibility requires this owner to be a subtype, not merely to call or store an API. '
    'Real type candidates are ranked from this responsibility and related source/type structure. '
    'more_symbols reads an actual successor page; inheritance can use only the inspected '
    'header_candidates in this page. Methods and fields are looked up while authoring their bodies. '
    'A superclass is a non-final class. An implemented type is an interface. The new owner is neither '
    'an inheritance candidate nor an implementation target in this phase. ' + STATE_INSTRUCTION)
MEMBER_INSTRUCTION = ('Plan only additional helper methods, fields or constructors. No Java source. '
    'Declarations with a body_owner already have an assigned author and will receive their actual bodies '
    'after this phase. Do not redeclare them. If no additional helper is needed, select begin_bodies '
    'with one queued body; every other assigned body remains queued and will also be authored. '
    'Use overload only to add a distinct signature to a helper, and declarations to read older members. '
    'Choose the helper name and purpose here; its types and dependency APIs are resolved in the following '
    'signature and body phases. Dependency search is not a member-planning action. ' + STATE_INSTRUCTION)
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
    header_policy = context['authoring_contract'].get('host_requirements', {}).get(
        'java_class_header', 'authored')
    if header_policy not in ('plain', 'authored'):
        raise ValueError('CONSTRUCTION_CLASS_HEADER_POLICY')
    context['source_pages'] = dict(context.get('source_pages', {}))
    from .construction_source_symbols import merge_materialized_source_sites
    symbols = merge_materialized_source_sites(
        context['source_pages'], context['symbols']
    )
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
        from .feature_work_order import FeatureWorkOrder
        requirement_id = str(bindings.get('parent_requirement_id')
                             or bindings.get('requirement_id')
                             or behaviors[0]['requirement_id'])
        work_order = FeatureWorkOrder.for_feature(
            requirement_id=requirement_id,
            target=context['target'],
            target_path=spec['target_path'],
            behaviors=behaviors,
        )
        # Both direct and legacy implementations receive the same immutable
        # requirement/target contract, so route decisions cannot erase effects.
        context['feature_work_order'] = work_order.to_dict()
        context['feature_work_order_sha256'] = work_order.fingerprint()
        retained_units = json.loads(bindings.get('retained_typed_behavior_json') or '{}')
        context['direct_method_forbidden'] = any(
            retained_units.get(key)
            for key in ('functions', 'event_bindings', 'ui_owned_records')
        )
        print('CONSTRUCTION FEATURE WORK ORDER: ' + json.dumps({
            'feature_id': work_order.feature_id,
            'target': work_order.to_dict()['target'],
            'behavior_count': len(work_order.behavior_ids),
            'route': 'fresh',
            'fingerprint': work_order.fingerprint(),
        }, ensure_ascii=False), flush=True)
    else:
        # Canonical leaves such as minecraft/entity/registry are not entrypoint
        # artifacts, yet they are the main source of one-operation-per-call
        # explosions. Give them the same whole-feature target/acceptance contract.
        from .feature_work_order import FeatureWorkOrder
        requirement_id = ','.join(task.requirement_ids)
        leaf_behavior = {
            'behavior_id': task.behavior_id or task.task_id,
            'requirement_id': requirement_id,
            'intent': task.intent,
            'effects': task.effects or task.intent,
            'state_reads': task.state_reads,
            'state_writes': task.state_writes,
        }
        work_order = FeatureWorkOrder.for_feature(
            requirement_id=requirement_id,
            target=context['target'],
            target_path=spec['target_path'],
            behaviors=[leaf_behavior],
        )
        context['feature_work_order'] = work_order.to_dict()
        context['feature_work_order_sha256'] = work_order.fingerprint()
        print('CONSTRUCTION FEATURE WORK ORDER: ' + json.dumps({
            'feature_id': work_order.feature_id,
            'target': work_order.to_dict()['target'],
            'behavior_count': 1, 'route': 'fresh',
            'fingerprint': work_order.fingerprint(),
        }, ensure_ascii=False), flush=True)
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
    # Default: let the model author a COMPLETE Java compilation unit, then
    # repair its actual compiler diagnostics. Do not silently fall back to
    # the historical node-by-node Typed IR on a failed candidate: that route
    # can indefinitely expand missing block inputs without executable Java.
    # The old path remains manually opt-in for controlled regression tests.
    import os
    if os.environ.get("MMM_LEGACY_TYPED_IR") != "1":
        # Host-owned Java declaration frames are the production default.
        # The former whole-file model author remains explicitly opt-in only;
        # never recover a failed member by switching back to unbounded imports.
        if os.environ.get("MMM_LEGACY_FILE_AUTHOR") == "1":
            from .construction_file_author import compose_complete_source as author
            mode = "legacy_file_level_java"
        else:
            from .construction_host_source import compose_host_java_source as author
            mode = "host_scaffold_java"
        print('CONSTRUCTION_AUTHOR_ROUTE: ' + json.dumps({
            'mode': mode, 'owner': owner,
            'assigned_behaviors': len(behaviors),
            'target_build_verified': False,
            'legacy_automatic_fallback': False,
        }, ensure_ascii=False), flush=True)
        return author(
            planning_router, spec, context, package=package, name=name, owner=owner,
            behaviors=behaviors, source_context=source_context,
        )
    # Explicitly selected legacy implementation, never an automatic fallback.
    from .construction_direct_writer import eligible as direct_eligible, compose_feature
    if direct_eligible(bindings, context, behaviors):
        print('CONSTRUCTION_AUTHOR_ROUTE: ' + json.dumps({
            'mode': 'direct_java', 'owner': owner,
            'assigned_behaviors': len(behaviors), 'checkpointed': True,
            'target_build_verified': False,
        }, ensure_ascii=False), flush=True)
        try:
            return compose_feature(router, spec, context, behaviors,
                                   package=package, name=name, owner=owner)
        except ValueError as exc:
            from .construction_source_repair import report_recoverable_direct_failure
            if not report_recoverable_direct_failure(owner, exc):
                raise
    if bindings.get('request_feature_entrypoint'):
        retained = json.loads(bindings.get('retained_typed_behavior_json') or '{}')
        print('CONSTRUCTION_AUTHOR_ROUTE: ' + json.dumps({
            'mode': 'typed_ir', 'owner': owner,
            'assigned_behaviors': len(behaviors),
            'stateful_behaviors': sum(bool(row.get('state_reads') or row.get('state_writes'))
                                       for row in behaviors),
            'retained_typed_effects': sum(len(retained.get(key, ())) for key in
                                          ('functions', 'event_bindings', 'ui_owned_records')),
            'owned_runtime_record_count': len(json.loads(
                bindings.get('owned_runtime_records_json') or '[]')),
        }, ensure_ascii=False), flush=True)
    interfaces, superclass = [], ''
    symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': name, 'access': 1,
        'superclass': superclass, 'interfaces': interfaces,
        'side': str(spec.get('side', 'COMMON')), 'provenance': 'construction_plan'}
    from .construction_retrieval import ConstructionEvidenceIndex
    context['evidence_index'] = ConstructionEvidenceIndex(symbols, context.get('source_pages', {}), router,
        target_path=task.target_path, owner=owner, authoring_contract=context['authoring_contract'])
    state = ConstructionState(symbols, owner, task.target_path,
        authoring_contract=context['authoring_contract'], binding_pages=context['binding_pages'],
        evidence_index=context['evidence_index'])
    payload = {'intent': task.intent, 'trigger': task.trigger, 'effects': task.effects,
               'state_reads': task.state_reads, 'state_writes': task.state_writes,
               'owner': owner, 'target_path': task.target_path,
               'authoring_contract': context['authoring_contract']}
    if interface:
        exact_type(interface, symbols, allow_void=False)
        interfaces.append(interface)
        payload['host_entrypoint'] = {'interface': interface, 'method': lifecycle}
        state.advance()
    elif header_policy == 'authored':
        state.enter('class_header')
        while True:
            state.prepare_symbols(task.intent, kinds=('CLASS',))
            candidates = header_candidates(state)
            schema = header_schema(state, candidates)
            actions = choice_actions(schema)
            choice = ask(planning_router, 'construction_class_next',
                {**payload, **state.payload(actions, header_candidates=candidates)}, schema, HEADER_INSTRUCTION)
            if choice['action'] == 'finish_header':
                break
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
    # A plain artifact is a code container with an implicit Object superclass;
    # dependency APIs stay available to the authored bodies, not its header.
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
    # Registrars already have behavior-owned method bodies. Do not repeatedly ask
    # the small planner to invent additional helpers before writing any Java.
    preferred_body = registrars[0][0] if registrars else ''
    # Registrars with persistent state may still need explicit fields; allow
    # at most one additional field slot per distinct documented state name.
    shared_state_names = {str(value) for behavior in behaviors if isinstance(behavior, dict)
                          for kind in ('state_reads', 'state_writes')
                          for value in behavior.get(kind, ()) if str(value).strip()}
    remaining_state_fields = len(shared_state_names) if registrars else 0
    parent_owner = superclass or 'java.lang.Object'
    parent_constructors = [key for key, row in symbols.items() if row.get('kind') == 'CONSTRUCTOR'
        and row['owner'] == parent_owner and state.evidence_index.accessible(key)]
    implicit_super_available = any(not invocation_parameter_types(symbols[key])
        and not any(not (state.evidence_index._subtype(type_, 'java.lang.RuntimeException')
                        or state.evidence_index._subtype(type_, 'java.lang.Error'))
                    for type_ in symbols[key].get('throws', ())) for key in parent_constructors)
    while not registrars or remaining_state_fields:
        state.enter('member_declarations')
        # The page is already bounded; every remaining body still stays queued.
        body_candidates = [key for key in state.declaration_ids if
                           symbols[key].get('body_owner') != 'host_lifecycle'][-8:]
        schema = member_schema(state, reserved_names=reserved_names, body_candidates=body_candidates)
        if registrars:
            # The shared member schema groups method and field in one enum.
            # Narrow that branch to field without accidentally deleting it.
            state_branches = []
            for branch in schema['oneOf']:
                action_spec = branch.get('properties', {}).get('action', {})
                if action_spec.get('const') == 'begin_bodies':
                    state_branches.append(branch)
                elif 'field' in action_spec.get('enum', ()):
                    state_branches.append({**branch, 'properties': {
                        **branch['properties'], 'action': {'const': 'field'}}})
            schema['oneOf'] = state_branches
        if not implicit_super_available and not any(symbols[key]['kind'] == 'CONSTRUCTOR' for key in state.declaration_ids):
            schema['oneOf'] = [branch for branch in schema['oneOf']
                               if branch.get('properties', {}).get('action', {}).get('const') != 'begin_bodies']
            payload['required_constructor_contract'] = {
                'superclass': parent_owner, 'constructors': [symbol_record(key, symbols[key]) for key in parent_constructors],
                'obligation': 'Declare a constructor and author its exact superclass invocation before bodies can finish.',
            }
        actions = choice_actions(schema)
        item = ask(planning_router, 'construction_member_next',
            {**payload, **state.payload(actions, body_candidates=body_candidates)}, schema, MEMBER_INSTRUCTION)
        action = item['action']
        if action == 'begin_bodies':
            preferred_body = item['symbol_ref']
            break
        if consume_lookup_choice(state, item):
            continue
        if action == 'declarations':
            state.lookup_declarations(item['offset'])
            continue
        member_name = name if action == 'constructor' else identifier(item['name'])
        if member_name in reserved_names:
            raise ValueError('CONSTRUCTION_HOST_DECLARATION_RESERVED')
        explicit_overload = action == 'overload'
        if explicit_overload:
            action = 'method'
        current = {**payload, 'kind': action, 'name': member_name, 'purpose': item['purpose']}
        # Fresh, uncalled helper methods have no external signature to infer.
        # The previous path let a 9B model invent arbitrary return types and
        # dozens of parameters before a single declaration was accepted.
        # Only a field or explicit overload needs a free type decision.
        return_type = (choose_type(planning_router, state, current,
            phase='member_return_type', allow_void=action != 'field')
            if action == 'field' or explicit_overload else 'void')
        state.enter('member_signature')
        schema = obj({**SIGNATURE['properties'], 'return_type': {'const': return_type}})
        if action != 'field':
            # The body author handles checked exceptions locally. Java permits
            # methods and overrides to declare no propagated exceptions.
            current['exception_contract'] = {
                'rule': 'Declare only intended propagated exceptions. Bodies must catch other checked exceptions; overrides cannot broaden inherited throws.',
                'inherited_methods': [symbol_record(key, row) for key, row in symbols.items()
                    if row.get('kind') == 'METHOD' and row['name'] == member_name
                    and row['owner'] != owner and state.evidence_index._subtype(owner, row['owner'])],
            }
        if action == 'constructor':
            schema['properties']['static'] = {'const': False}
            schema['properties']['final'] = {'const': False}
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
        if action != 'field':
            # Avoid an unbounded exception-enumeration loop. Checked exceptions
            # must be caught by the generated body, not guessed as declarations.
            metadata['throws'] = []
        state.advance()
        parameters = []
        # New helper methods have no incoming Java contract: their first
        # implementation should take no arguments and use inspected APIs in
        # the body. Constructors/explicit overloads retain signature authoring.
        if action == 'constructor' or explicit_overload:
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
        if registrars and action == 'field':
            remaining_state_fields -= 1
        if action in {'method', 'constructor'}:
            # A small model needs a complete executable unit before expanding
            # the surrounding class. Queue this body immediately instead of
            # generating an unlimited catalogue of speculative helpers.
            preferred_body = key
            break
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

    context['construction_declaration_ids'] = tuple(state.declaration_ids)
    from .construction_progress import planned_declarations_fingerprint, construction_input_fingerprint
    planned_declaration_hash = planned_declarations_fingerprint(symbols, state.declaration_ids)
    input_hash = construction_input_fingerprint(spec, context['authoring_contract'])
    member_invocations = {}
    constructor_delegations = {}
    accepted_member_evidence = {}

    def author_one(member_task, *, owned_records=()):
        from .construction_progress import ConstructionNodeCheckpoint
        if not context.get('project_root'):
            raise ValueError('CONSTRUCTION_RESUME_WORKSPACE_REQUIRED')
        progress = ConstructionNodeCheckpoint(context['project_root'], member_task,
            target=context['target'],
            prerequisite_hashes=context['prerequisite_hashes'],
            declaration=symbols[member_task.owner_symbol], owner_header=symbols[owner],
            planned_declaration_hash=planned_declaration_hash,
            construction_input_hash=hashlib.sha256(json.dumps(
                {'input': input_hash, 'accepted_sibling_evidence': accepted_member_evidence},
                ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        scoped_context = dict(context)
        scoped_context['owned_runtime_records'] = list(owned_records)
        scoped_context['source_read'] = progress.record_source_read
        scoped_context['already_implemented_effects'] = {
            'ui_owned_records': typed_owners.get('ui_owned_records', ()),
            'typed_functions': typed_owners.get('functions', ()),
            'typed_events': typed_owners.get('event_bindings', ()),
        }
        scoped_context['resume_nodes'] = lambda current: progress.load()
        scoped_context['checkpoint_transaction'] = lambda current, accepted, stage, snapshot, expected=None: progress.save(
            current, accepted, stage, author_state=snapshot, expected_graph_sha256=expected)
        scoped_context['resume_author_state'] = progress.load_author_state
        scoped_context['resume_node_purposes'] = lambda current, accepted: progress.load_author_state(
            current, accepted, purposes_only=True)
        scoped_context['checkpoint_author_state'] = progress.save_author_state
        scoped_context['recorded_source_refs'] = progress.recorded_source_refs
        nodes = author_task_operations(router, member_task, scoped_context)
        member_invocations[member_task.owner_symbol] = {
            node['attributes']['symbol_ref']
            for node in nodes
            if node['op'] in {'call', 'new', 'field', 'constructor_call'}
            and node['attributes'].get('symbol_ref')
        }
        member_invocations[member_task.owner_symbol].update(
            symbols[key]['inherited_from'] for key in tuple(member_invocations[member_task.owner_symbol])
            if symbols[key].get('inherited_from'))
        for node in nodes:
            if node['op'] == 'constructor_call' and node['attributes']['target'] == 'this':
                constructor_delegations[member_task.owner_symbol] = node['attributes']['symbol_ref']
        return nodes

    body_jobs = []
    for index, (key, purpose) in enumerate(declarations):
        member_task = replace(task, task_id=task.task_id + '.member.' + str(index), owner_symbol=key,
                              intent=purpose, output_symbols=(key,))
        body_jobs.append((member_task, {}))
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
    from .construction_member_direct_author import eligible_method, compose_method
    for member_task, options in authoring_order:
        direct_candidate = eligible_method(member_task, symbols, context, **options)
        nodes = ()
        if direct_candidate:
            try:
                # The direct route is fast only when its *real* classpath types
                # resolve. A syntactically valid invented API is not accepted.
                rendered[member_task.owner_symbol] = compose_method(
                    router, member_task, context, owner_name=name)
                member_invocations[member_task.owner_symbol] = set()
            except ValueError as exc:
                from .construction_source_repair import report_recoverable_direct_failure
                if not report_recoverable_direct_failure(member_task.owner_symbol, exc):
                    raise
                direct_candidate = False
        if not direct_candidate:
            # The fallback is not free-form Java regeneration. Every operation
            # is selected from inspected signatures and validated for input
            # types before the host renders source from the typed IR.
            nodes = author_one(member_task, **options)
            render_task_java(member_task, nodes, symbols)
            rendered[member_task.owner_symbol] = render_task_member(member_task, nodes, symbols)
        print('CONSTRUCTION_MEMBER_PROGRESS: ' + json.dumps({
            'owner_symbol': member_task.owner_symbol,
            'accepted_operation_count': len(nodes),
            'rendered_java_bytes': len(rendered[member_task.owner_symbol].encode('utf-8')),
            'target_compiled': False,
        }, ensure_ascii=False), flush=True)
        # A following member must be able to inspect decisions already accepted
        # here (state encodings, callbacks and helper logic), not only signatures.
        ref = 'accepted-member:' + hashlib.sha256(member_task.owner_symbol.encode()).hexdigest()[:24]
        evidence_hash = hashlib.sha256(json.dumps(
            {'task': member_task.to_dict(), 'nodes': nodes,
             'rendered_java_sha256': hashlib.sha256(
                 rendered[member_task.owner_symbol].encode('utf-8')).hexdigest()},
            ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        metadata = {'owner': owner, 'authority': 'accepted_sibling_graph',
                    'source_sha256': evidence_hash, 'side': symbols[member_task.owner_symbol]['side']}
        # Make the *actual rendered Java* retrievable for the next member.
        # The graph remains canonical; this is read-only sibling evidence,
        # never an assertion that the source has already compiled or run.
        units = [{**metadata, 'kind': 'accepted_member', 'signature': member_task.owner_symbol,
                  'code': json.dumps({'task': member_task.to_dict(), 'node_count': len(nodes),
                                      'node_catalogue': {'source_ref': ref, 'first_node_offset': 2,
                                                         'unit_count': len(nodes)}}, ensure_ascii=False)},
                 {**metadata, 'kind': 'accepted_java_member',
                  'authority': 'rendered_sibling_not_yet_target_compiled',
                  'signature': member_task.owner_symbol,
                  'code': rendered[member_task.owner_symbol]}]
        units.extend({**metadata, 'kind': 'accepted_node',
                      'signature': member_task.owner_symbol + '::' + row['node_id'] + ' ' + member_task.intent,
                      'code': json.dumps(row, ensure_ascii=False)} for row in nodes)
        context['source_pages'][ref] = units
        context['evidence_index'].include_source(ref, units)
        accepted_member_evidence[ref] = evidence_hash
    # Keep declaration/field initialization order independent of author order.
    members = [rendered[current.owner_symbol] for current, _options in body_jobs]
    checked_delegations = set()
    for start in constructor_delegations:
        current, visiting = start, set()
        while current in constructor_delegations and current not in checked_delegations:
            if current in visiting:
                raise ValueError('CONSTRUCTION_CONSTRUCTOR_DELEGATION_CYCLE')
            visiting.add(current)
            current = constructor_delegations[current]
        checked_delegations.update(visiting)
    if registrars:
        lifecycle_calls = '\n'.join('this.' + symbols[key]['name'] + '();'
                                    for key, _ in registrars)
        members.append('@Override\npublic void onInitialize() {\n' + lifecycle_calls + '\n}')
    source = 'package ' + package + ';\n\npublic ' + ('final ' if lifecycle else '') + 'class ' + name
    source += (' extends ' + superclass if superclass else '') + (' implements ' + ', '.join(interfaces) if interfaces else '')
    source += ' {\n' + '\n\n'.join(members) + '\n}\n'
    from .compiled_feature_patch import replay_compiled_repair
    source = replay_compiled_repair(context['project_root'], spec['target_path'], source)
    if context.get('verify_source'):
        context['verify_source'](source, spec['target_path'])
    from .construction_coverage import record_authored_source_calls
    record_authored_source_calls(context['project_root'], spec['target_path'], source,
)
    return source
