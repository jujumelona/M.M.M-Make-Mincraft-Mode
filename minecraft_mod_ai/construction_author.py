"""One bounded typed node per inference; host graph/inventory never enter prompts."""
from __future__ import annotations
import json
from copy import deepcopy
from itertools import islice
from .java_operation_ir import ATTRIBUTES, REQUIRED, JavaOperations
from .fixed_template_generation import _validate_native_arguments
from .construction_state import (ConstructionState, STATE_INSTRUCTION, PAGE_SIZE,
                                 page_result, symbol_record)

TEXT = {'type': 'string', 'maxLength': 240}
ID = {'type': 'string', 'maxLength': 240, 'minLength': 1}

def obj(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required, 'additionalProperties': False}

def array(items, maximum=None):
    return {'type': 'array', 'items': items, **({'maxItems': maximum} if maximum is not None else {})}

def ask(router, name, payload, schema, instruction):
    from .model_concurrency import raise_if_model_execution_cancelled
    raise_if_model_execution_cancelled()
    messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    decision = _validate_native_arguments(router.generate_structured_data('coder', messages, data_name=name, parameters=schema), schema)
    raise_if_model_execution_cancelled()
    state = payload.get('construction_state', {})
    print('CONSTRUCTION DECISION: ' + json.dumps({
        'decision': name, 'phase': state.get('phase'), 'revision': state.get('revision'),
        'declaration_count': state.get('declarations', {}).get('total_count'),
        'accepted_node_count': state.get('graph', {}).get('accepted_node_count'),
        'lookups_since_progress': state.get('lookups_since_progress'),
        'choice': {key: decision[key] for key in
                   ('action', 'complete', 'query', 'offset', 'name', 'symbol_ref', 'type_ref', 'base_type',
                    'port', 'behavior_id', 'node_id', 'purpose')
                   if key in decision},
    }, ensure_ascii=False), flush=True)
    return decision

def operation_schema(op, *, node_ids=()):
    from .construction_declaration_choices import identifier_schema, string_excluding
    from .construction_operation_choices import NODE_ID_ALPHABET
    scalar = {'anyOf': [{'type': 'string', 'maxLength': 240}, {'type': 'number'}, {'type': 'boolean'}, {'type': 'null'}]}
    node_id = string_excluding(node_ids, alphabet=NODE_ID_ALPHABET)
    node_ref = string_excluding((), alphabet=NODE_ID_ALPHABET)
    attributes = {'value': scalar, 'name': identifier_schema(), 'symbol_ref': ID, 'type_ref': ID, 'element_type': ID,
        'operator': {'type': 'string', 'maxLength': 4}, 'final': {'type': 'boolean'}, 'sized': {'type': 'boolean'},
        'target': {'enum': ['this', 'super']},
        'finally': {'type': 'boolean'}, 'parameters': array(identifier_schema()),
        'selector_type': ID, 'cases': array({'oneOf': [obj({'kind': {'const': 'default'}}),
                                                    obj({'kind': {'const': 'value'}, 'value': scalar})]}),
        'catches': array(obj({'name': identifier_schema(), 'type_ref': ID}))}
    return obj({'node_id': node_id, 'input_refs': array(node_ref), 'typed_result': ID,
                'attributes': obj({key: attributes[key] for key in sorted(ATTRIBUTES[op])}, sorted(REQUIRED[op]))})


def _node_summary(row):
    return {k: row[k] for k in ('node_id', 'op', 'typed_result', 'input_refs')} | {
        'names': {k: v for k, v in row['attributes'].items() if k in {'name', 'parameters'}}}


def _graph_refs(nodes):
    by_id = {row['node_id']: row for row in nodes}
    referenced = {ref for row in nodes for ref in row['input_refs']}
    pending = dict.fromkeys(ref for row in nodes for ref in row['input_refs'] if ref not in by_id)
    roots = dict.fromkeys(key for key in by_id if key not in referenced)
    return by_id, referenced, pending, roots


def _has_observable_effect(nodes, owner_symbol):
    # This source-only check is shared by resumed and in-progress bodies.
    return any(
        row['op'] in {'assign', 'constructor_call'} or
        (row['op'] == 'call' and row['attributes']['symbol_ref'] != owner_symbol)
        for row in nodes
    )


def author_task_operations(router, task, context):
    # Choice helpers import the shared schema primitives above; load them only
    # after this module is initialized, avoiding a declaration/author cycle.
    from .construction_operation_choices import operation_next_schema, operation_context_branches, MAX_OPERATION_EVIDENCE_READS
    from .construction_declaration_choices import choice_actions
    from .construction_operation_binding import bind_operation
    from .construction_operation_contracts import named_values, pending_input_contracts
    if not callable(context.get('checkpoint_transaction')):
        raise ValueError('CONSTRUCTION_CHECKPOINT_TRANSACTION_REQUIRED')
    symbols = context['symbols']
    declaration = symbols.get(task.owner_symbol)
    if declaration is None:
        raise ValueError('CONSTRUCTION_SYMBOL_MISSING: ' + task.owner_symbol)
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
    by_id, referenced, pending, roots = _graph_refs(nodes)
    if len(by_id) != len(nodes):
        raise ValueError('CONSTRUCTION_CHECKPOINT_NODES_INVALID')
    if stage == 'validated':
        JavaOperations(task, nodes, symbols)
        # Legacy validated graph checkpoints are not proof of the requested
        # gameplay effect. Prevent old return-only bodies bypassing this gate.
        if task.behavior_id and not _has_observable_effect(nodes, task.owner_symbol):
            raise ValueError('CONSTRUCTION_BEHAVIOR_EFFECT_MISSING: ' + task.task_id)
        return tuple(nodes)
    from .construction_feedback import graph_digest, replace_node, rejection, admission
    validator = JavaOperations(task, (), symbols, partial=True)
    validation = validator.validate(nodes)
    declaration_ids = context.get('construction_declaration_ids')
    if declaration_ids is None:
        declaration_ids = sorted(key for key, row in symbols.items()
                                 if row.get('owner') == declaration['owner'] and row.get('kind') != 'CLASS')
    source_units = context.get('source_pages', {})
    evidence_index = context.get('evidence_index')
    if evidence_index is None:
        from .construction_retrieval import ConstructionEvidenceIndex
        evidence_index = ConstructionEvidenceIndex(symbols, source_units, router,
            target_path=task.target_path, owner=declaration['owner'],
            authoring_contract=base['authoring_contract'])
    state = ConstructionState(symbols, declaration['owner'], task.target_path,
        declaration_ids=declaration_ids, revision=len(nodes),
        authoring_contract=context.get('authoring_contract'), binding_pages=context.get('binding_pages'),
        evidence_index=evidence_index)
    state.enter('operation_graph')
    source_ids = sorted(source_units)
    read_refs = context.get('recorded_source_refs')
    read_sources = set(read_refs() if callable(read_refs) else ())

    def lookup_result(action, query, offset):
        if action == 'symbols':
            return state.symbol_result(query, offset)
        if action == 'scopes':
            return evidence_index.scope_page(query, offset)
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
            if offset != 0:
                raise ValueError('CONSTRUCTION_LOOKUP_OFFSET_INVALID')
            if len(keys) > PAGE_SIZE or any(key not in by_id for key in keys):
                raise ValueError('CONSTRUCTION_NODE_REFERENCE_MISSING')
            return {'nodes': [by_id[key] for key in keys], 'total_count': len(keys),
                    'status': 'matches', 'next_offset': None}
        if action == 'sources':
            return evidence_index.source_page(query, offset)
        if action == 'bindings':
            return state.binding_result(query, offset)
        if action != 'source' or query not in source_units:
            raise ValueError('CONSTRUCTION_SOURCE_PAGE_MISSING')
        pages = source_units[query]
        if offset >= len(pages):
            raise ValueError('CONSTRUCTION_SOURCE_PAGE_OFFSET_INVALID')
        return {'source': pages[offset:offset + 1], 'total_count': len(pages),
                'status': 'matches', 'next_offset': offset + 1 if len(pages) > offset + 1 else None}

    resume_author = context.get('resume_author_state')
    saved_state = resume_author(task, tuple(nodes)) if callable(resume_author) else None
    if saved_state is not None:
        last = state.restore(saved_state, resolve_lookup=lookup_result)
        if last is not None:
            state.lookup = {**lookup_result(last['kind'], last['query'], last['offset']), **last}
    else:
        recover_purposes = context.get('resume_node_purposes')
        if callable(recover_purposes):
            for key, purpose in (recover_purposes(task, tuple(nodes)) or {}).items():
                state.remember_node_purpose(key, purpose)

    def save_author():
        callback = context.get('checkpoint_author_state')
        if callable(callback):
            callback(task, tuple(nodes), state.export())

    def read_lookup(choice):
        # Memory navigation is a model decision too. Counting only symbol
        # requests allowed infinite recall/index cycles with zero Java nodes.
        if state.phase == 'operation_graph':
            state.operation_reads += 1
        action = choice['action']
        if action == 'memory_index':
            state.memory.select_index(choice['offset'])
            return
        if action == 'recall':
            state.memory.recall(choice['evidence_id'])
            return
        if action == 'more_symbols':
            action = 'symbols'
        query = choice.get('query', '').strip()
        offset = choice.get('offset', 0)
        if action == 'scopes':
            state.lookup_scopes(query, offset)
        else:
            state.consume_lookup(action, query, offset, lookup_result(action, query, offset))
        if action == 'source':
            callback = context.get('source_read')
            if callable(callback):
                callback(task, query)
            read_sources.add(query)

    def reject_candidate(error, *, candidate=None, phase='partial'):
        observation = rejection(error, nodes, candidate=candidate, phase=phase)
        state.validation_observation = observation
        state.reconsider_operation()
        state.operation_rejections = {}
        target = observation['node_id']
        if target in by_id:
            state.selection_goal = {'node_id': target,
                'purpose': state.node_purposes.get(target, task.intent), 'excluded_operations': []}
        elif candidate is None:
            state.selection_goal = None
        else:
            state.selection_goal['excluded_operations'] = []
        print('CONSTRUCTION OPERATION OBSERVATION: ' + json.dumps({
            'status': 'rejected', 'phase': phase, 'node_id': target,
            'diagnostic': observation['diagnostic'], 'graph_changed': False,
        }, ensure_ascii=False), flush=True)
        save_author()

    def checkpoint_graph(stage, expected=None):
        context['checkpoint_transaction'](task, tuple(nodes), stage, state.export(), expected)

    if base['port_source_ref'] and base['port_source_ref'] not in read_sources:
        read_lookup({'action': 'source', 'query': base['port_source_ref'], 'offset': 0})
    save_author()

    instruction = ('Author this current behavior as typed Java operations, one node per turn. No Java source strings. '
        'selection_goal identifies the current missing input or root slot and its implementation purpose. '
        'Choose the Java operation from the required behavior and known values first. '
        'Read configuration or source only for a specific missing fact. API and type lookup happens '
        'after the operation is selected; do not browse APIs to decide whether to implement it. '
        'Exact API names come from returned declarations. Context reads keep '
        'the same slot and purpose; when the facts are sufficient, choose its Java operation. '
        'State the operation purpose in plain language, then bind the correct actual API before writing its node. '
        'API signatures fix result types and receiver/argument counts. Required inputs may be forward '
        'references, and each missing input must be authored. Relevant graph nodes are stored on the '
        'host; do not recreate them. '
        'Implement only owned runtime records; never duplicate effects already provided by Typed/UI. '
        'When a pinned donor source is installed, use its exact inspected callable symbols '
        'through typed call/field/new operations and implement missing state and event glue. '
        'operation_observation is the actual result of the previous operation. A rejected node '
        'was not admitted: correct its diagnostic in the same slot. If an earlier provisional '
        'parent is the cause, revise that node while preserving independent work. revise_node '
        'selects a visible existing node for a local replacement. Partial admission proves only '
        'closed type/access constraints, not the requested gameplay effect. '
        'Only the latest four node summaries are shown. Use node_index to discover older node IDs, '
        'then nodes to read one complete known node. Use sources to discover source IDs, then source '
        'to read one complete unit. Use bindings with a binding_id from authoring_contract.binding_pages '
        'to read complete configuration records. Respect the canonical responsibility, exact identities '
        'and implementation_requirements in authoring_contract. Forward references are allowed; pending_input_refs must be authored '
        'before complete. Finish only after all effects and return paths are represented. '
        'Only the supported typed operations can be authored; the host validates operation/type/access '
        'and complete behavior contracts before rendering. No raw Java or placeholder success. ' + STATE_INSTRUCTION)
    while True:
        observation = state.validation_observation or {}
        if observation and observation['graph_sha256'] != graph_digest(nodes):
            raise ValueError('CONSTRUCTION_OBSERVATION_GRAPH_MISMATCH')
        repair_id = (observation.get('node_id') if observation.get('status') == 'rejected'
                     and observation.get('node_id') in by_id else None)
        working_nodes = nodes
        contracts = pending_input_contracts(nodes, declaration, symbols, replacing=repair_id)
        if state.selection_goal is None:
            if repair_id:
                slot = repair_id
            elif pending:
                slot = next(iter(pending))
            else:
                index = len(nodes)
                slot = 'operation_' + str(index)
                while slot in by_id:
                    index += 1
                    slot = 'operation_' + str(index)
            parts = []
            if slot in state.node_purposes:
                parts.append(state.node_purposes[slot])
            elif not contracts.get(slot):
                parts.extend((task.intent, declaration.get('purpose', '')))
            for consumer in contracts.get(slot, ()):
                parts.extend(consumer.get(key, '') for key in ('expected_type', 'expected_element_type'))
                parts.extend(consumer.get('expected_types', ()))
                parts.extend(consumer.get('allowed_types', ()))
                parent = by_id[consumer['consumer']]
                if slot not in state.node_purposes:
                    parts.append('Input ' + str(consumer['input_index']) + ' for ' +
                                 state.node_purposes.get(parent['node_id'], task.intent))
                if parent['op'] in {'call', 'new', 'field'}:
                    api = symbols[parent['attributes']['symbol_ref']]
                    parts.append(api.get('purpose') or api['owner'] + '.' + api['name'])
            purpose = '\n'.join(dict.fromkeys(part for part in parts if part))
            state.selection_goal = {'node_id': slot, 'purpose': purpose, 'excluded_operations': []}
            # Never pre-browse the entire dependency inventory before the model
            # chooses call/new/field/control flow. API lookup happens in binding.
            save_author()
        slot = state.selection_goal['node_id']
        if slot != repair_id and (slot in by_id or (pending and slot not in pending)):
            raise ValueError('CONSTRUCTION_SELECTION_GOAL_NODE_INVALID')
        recent = [_node_summary(row) for row in nodes[-4:]]
        # A gameplay behavior must not certify a syntactically valid no-op.
        # Actual runtime verification remains separate, but a body containing
        # only return/literal/block cannot fulfill an observable effect.
        observed_effect_node = _has_observable_effect(nodes, task.owner_symbol)
        schema = operation_next_schema(state, nodes=working_nodes, declaration=declaration,
            can_complete=bool(nodes) and not pending and not repair_id
                         and observation.get('status') != 'rejected'
                         and (not task.behavior_id or observed_effect_node),
            context_branches=operation_context_branches(state, nodes=nodes,
                source_units=source_units, pinned_source=base['port_source_ref'],
                resolve_symbols=False))
        actions = choice_actions(schema)
        payload = {**base, 'recent_nodes': recent, **state.payload(actions,
            current_member=task.owner_symbol,
            graph={'accepted_node_count': len(nodes), 'resumed_node_count': len(existing),
                   'checkpoint_stage': stage or 'new',
                   'validated_node_count': len(validation.validated_node_ids),
                   'deferred_node_count': len(validation.deferred_node_ids),
                   'behavior_verified': False, 'repair_node_id': repair_id,
                   'pending_input_ref_count': len(pending), 'pending_input_refs': list(islice(pending, PAGE_SIZE)),
                   'root_count': len(roots), 'root_ids': list(islice(roots, PAGE_SIZE))},
            sources={'total_count': len(source_ids), 'recent_ids': source_ids[:PAGE_SIZE],
                     'lookup_action': 'sources',
                     'pinned_source_read': bool(base['port_source_ref'] in read_sources)})}
        payload['pending_input_contracts'] = {key: contracts[key] for key in list(contracts)[:PAGE_SIZE]}
        payload['named_values'] = named_values(working_nodes, declaration, symbols, node_id=slot)
        choice = state.operation_goal or ask(router, 'construction_operation_next', payload, schema, instruction)
        action = choice['action']
        if action in {'symbols', 'more_symbols', 'scopes', 'bindings', 'source', 'sources',
                      'declarations', 'nodes', 'node_index', 'memory_index', 'recall'}:
            read_lookup(choice)
            save_author()
            continue
        if action == 'revise_node':
            state.validation_observation['node_id'] = choice['node_ref']
            state.selection_goal = {'node_id': choice['node_ref'], 'purpose': choice['purpose'],
                                    'excluded_operations': []}
            state.reconsider_operation()
            state.operation_rejections = {}
            save_author()
            continue
        if action == 'complete':
            try:
                validator.validate(nodes, partial=False)
            except ValueError as error:
                reject_candidate(error, phase='complete')
                continue
            checkpoint_graph('validated')
            return tuple(nodes)
        if max(len(state.requests), state.operation_reads) >= MAX_OPERATION_EVIDENCE_READS:
            # Migrate in-progress checkpoints captured during the historic
            # 240-page browsing loop to a fresh, operation-specific window.
            # Keep the selected operation and accepted nodes unchanged.
            state.reconsider_operation()
            print('CONSTRUCTION EVIDENCE WINDOW RESET: ' + json.dumps({
                'node_id': slot, 'operation': action,
                'reason': 'legacy_or_exhausted_preselection_navigation',
            }, ensure_ascii=False), flush=True)
        state.operation_goal = dict(choice)
        if choice['node_id'] != repair_id and (choice['node_id'] in by_id or (pending and choice['node_id'] not in pending)):
            raise ValueError('CONSTRUCTION_OPERATION_GOAL_NODE_INVALID')
        try:
            row = bind_operation(router, state, goal=choice, payload=payload, nodes=working_nodes,
                source_units=source_units, pinned_source=base['port_source_ref'],
                read_lookup=read_lookup, instruction=instruction, save_author=save_author)
        except ValueError as error:
            if not str(error).startswith(('CONSTRUCTION_OPERATION_NAMED_VALUE_MISSING',
                                          'CONSTRUCTION_OPERATION_EVIDENCE_MISSING')):
                raise
            reject_candidate(error, candidate={'node_id': slot, 'op': action,
                'input_refs': [], 'attributes': {}, 'purpose': choice['purpose']})
            continue
        if row is None:
            # The chosen API-dependent operation lacked verified callable
            # evidence. Reconsider the same slot rather than endlessly paging
            # symbols or fabricating a Java registry method. Persist exclusions
            # so resuming cannot repeat the same dead end.
            excluded = state.selection_goal.setdefault('excluded_operations', [])
            if action not in excluded:
                excluded.append(action)
            print('CONSTRUCTION OPERATION RECONSIDER: ' + json.dumps({
                'node_id': slot, 'rejected_operation': action,
                'excluded_operations': excluded,
                'reason': 'no_usable_api_evidence',
            }, ensure_ascii=False), flush=True)
            state.reconsider_operation()
            save_author()
            continue
        row['op'] = action
        if row['node_id'] in by_id and row['node_id'] != repair_id:
            raise ValueError('CONSTRUCTION_NODE_DUPLICATE')
        try:
            candidate_nodes = replace_node(nodes, row) if repair_id else (*nodes, row)
            candidate_validation = validator.validate(candidate_nodes)
        except ValueError as error:
            reject_candidate(error, candidate=row)
            continue
        expected = graph_digest(nodes) if repair_id else None
        previous_ids = set(by_id)
        state.remember_node_purpose(row['node_id'], choice['purpose'], replace=True)
        nodes = list(candidate_nodes)
        by_id, referenced, pending, roots = _graph_refs(nodes)
        if repair_id:
            state.memory.invalidate_nodes((previous_ids - by_id.keys()) | {repair_id})
        live = set(by_id) | referenced
        state.node_purposes = {key: purpose for key, purpose in state.node_purposes.items() if key in live}
        state.advance(preserve_symbols=True)
        state.revision = len(nodes)
        state.validation_observation = admission(nodes, candidate_validation, row['node_id'])
        validation = candidate_validation
        stage = 'authoring'
        checkpoint_graph('authoring', expected)
        print('CONSTRUCTION OPERATION OBSERVATION: ' + json.dumps({
            'status': 'admitted', 'node_id': row['node_id'], 'revised': bool(repair_id),
            'validated_node_count': len(validation.validated_node_ids),
            'deferred_node_count': len(validation.deferred_node_ids), 'behavior_verified': False,
        }, ensure_ascii=False), flush=True)
        save_author()
