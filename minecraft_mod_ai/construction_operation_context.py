"""Exact local operation facts, with model-authored purposes kept as claims."""
from __future__ import annotations

from copy import deepcopy

from .construction_state import PAGE_SIZE, symbol_record
from .construction_type_contracts import owner_package
from .java_operation_ir import invocation_parameter_types, parameter_types


API_KINDS = {'call': 'METHOD', 'new': 'CONSTRUCTOR', 'field': 'FIELD',
             'constructor_call': 'CONSTRUCTOR'}


def api_input_slots(op, row, owner, symbols):
    """Use the same receiver and invocation order as the Java IR validator."""
    types = ([] if op == 'field' else invocation_parameter_types(row)
             if op in {'new', 'constructor_call'} else parameter_types(row))
    slots = []
    if op not in {'new', 'constructor_call'} and not row.get('static'):
        receiver = row['owner']
        if (row.get('access', 0) & 4
                and owner_package(symbols, row.get('declaring_owner') or receiver)
                != owner_package(symbols, owner)):
            receiver = owner
        slots.append({'input_index': 0, 'role': 'receiver', 'expected_type': receiver})
    enclosing = bool(op in {'new', 'constructor_call'} and row.get('enclosing_instance_type'))
    for index, type_ in enumerate(types):
        slot = {'input_index': len(slots), 'role': 'argument',
                'argument_index': index, 'expected_type': type_}
        if enclosing and index == 0:
            slot['enclosing_instance'] = True
        slots.append(slot)
    return slots


def api_use_claim(op, choice, *, author='model'):
    return {'operation': op, 'symbol_ref': choice['symbol_ref'],
            'intended_contribution': choice['intended_contribution'],
            'input_purposes': list(choice['input_purposes']),
            'author': author, 'semantic_status': 'unverified'}


def valid_api_use(claim, symbols, owner):
    """Check stored claim shape and actual signature arity, never its semantics."""
    if (not isinstance(claim, dict)
            or set(claim) != {'operation', 'symbol_ref', 'intended_contribution',
                              'input_purposes', 'author', 'semantic_status'}
            or not isinstance(claim.get('operation'), str)
            or claim.get('operation') not in API_KINDS
            or not isinstance(claim.get('symbol_ref'), str)
            or claim.get('symbol_ref') not in symbols
            or not isinstance(claim.get('author'), str)
            or claim.get('author') not in {'model', 'host_static_provider'}
            or claim.get('semantic_status') != 'unverified'):
        return False
    row = symbols[claim['symbol_ref']]
    purposes = claim['input_purposes']
    if (claim['author'] == 'host_static_provider'
            and (claim['operation'] != 'field' or not (row.get('static') or row.get('access', 0) & 8))):
        return False
    return (row.get('kind') == API_KINDS[claim['operation']]
            and isinstance(purposes, list)
            and len(purposes) == len(api_input_slots(claim['operation'], row, owner, symbols))
            and all(isinstance(value, str) and value.strip() and len(value) <= 240
                    for value in [claim['intended_contribution'], *purposes]))


def slot_context(state, nodes, declaration, contracts):
    """Expose the real consumers of this value, rather than only their types."""
    slot = state.selection_goal['node_id']
    by_id = {node['node_id']: node for node in nodes}
    consumers = []
    for contract in contracts.get(slot, ()):
        consumer = by_id[contract['consumer']]
        row = {**contract, 'consumer_purpose': state.node_purposes.get(consumer['node_id'], '')}
        symbol = consumer['attributes'].get('symbol_ref')
        if consumer['op'] in API_KINDS and symbol in state.symbols:
            api = state.symbols[symbol]
            row['selected_api'] = symbol_record(symbol, api)
            inputs = api_input_slots(consumer['op'], api, declaration['owner'], state.symbols)
            if contract['input_index'] < len(inputs):
                row['signature_input'] = inputs[contract['input_index']]
            claim = state.node_api_uses.get(consumer['node_id'])
            if claim:
                row['api_use_claim'] = deepcopy(claim)
        consumers.append(row)
    return {'node_id': slot, 'local_purpose': state.selection_goal['purpose'],
            'purpose_is_semantically_verified': False, 'consumers': consumers}


def body_progress(state, nodes, validation, *, can_complete):
    """Show what existing roots actually do before the model adds another."""
    by_id = {node['node_id']: node for node in nodes}
    referenced = {ref for node in nodes for ref in node['input_refs']}
    roots = [node for node in nodes if node['node_id'] not in referenced]
    validated = set(validation.validated_node_ids)

    def view(node):
        result = {key: deepcopy(node[key]) for key in ('node_id', 'op', 'typed_result')}
        result['purpose'] = state.node_purposes.get(node['node_id'], '')
        result['structural_status'] = 'typed' if node['node_id'] in validated else 'deferred'
        result['attributes'] = deepcopy(node['attributes'])
        symbol = node['attributes'].get('symbol_ref')
        if symbol in state.symbols:
            result['selected_api'] = symbol_record(symbol, state.symbols[symbol])
        if node['node_id'] in state.node_api_uses:
            result['api_use_claim'] = deepcopy(state.node_api_uses[node['node_id']])
        return result

    records = []
    for node in roots[-PAGE_SIZE:]:
        root = view(node)
        root['input_ref_count'] = len(node['input_refs'])
        root['inputs'] = [{'input_index': index, **(view(by_id[ref]) if ref in by_id else
                          {'node_id': ref, 'structural_status': 'missing'})}
                          for index, ref in enumerate(node['input_refs'][:PAGE_SIZE])]
        root['inspect_node_ref'] = node['node_id']
        records.append(root)
    return {'root_count': len(roots), 'recent_roots': records,
            'completion_available': can_complete, 'behavior_verified': False,
            'older_work_lookup': 'node_index followed by nodes',
            'meaning': 'These are actual accepted operations and literal values, not verified task effects. '
                       'Review them before adding a root. Repair an existing wrong value or strategy '
                       'instead of appending the same intended effect again. Repeated API use may be '
                       'valid for different inputs; API identity alone does not prove duplication. '
                       'Inspect a node to read its complete inputs; no older work was removed.'}


def task_context_bindings(context):
    """Keep broad task records exactly addressable through existing bindings."""
    pages, pointers = {}, []
    for field in ('owned_runtime_records', 'already_implemented_effects', 'donor_call_obligations'):
        value = context.get(field, ()) if field == 'owned_runtime_records' else context.get(field, {})
        if not value:
            continue
        if isinstance(value, dict):
            records = [{'key': key, 'value': deepcopy(item)} for key, item in value.items()]
        else:
            records = [deepcopy(item) for item in value]
        binding_id = 'operation-task:' + field
        pages[binding_id] = records
        pointers.append({'binding_id': binding_id, 'unit_count': len(records),
                         'origin': 'host_task_context.' + field})
    return pages, pointers


def focus_selected_api_memory(state, symbol_ref):
    """Writing needs the chosen signature; other exact APIs remain recallable."""
    state.memory.active = [key for key in state.memory.active
                           if state.memory.records[key]['selector']['kind'] not in {'symbols', 'declarations'}
                           or state.memory.records[key]['record']['symbol_id'] == symbol_ref]


def focused_operation_payload(payload):
    """Represent repeated signature evidence by reference, without losing facts."""
    result = deepcopy(payload)
    if result.get('operation_context_phase') == 'write':
        result.pop('api_candidates', None)
    signatures = {}

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if 'symbol_id' in value and 'kind' in value:
            reference = value['symbol_id']
            # Preserve distinct views if a record carries additional exact facts.
            suffix = 1
            while reference in signatures and signatures[reference] != value:
                reference = value['symbol_id'] + ':view_' + str(suffix)
                suffix += 1
            signatures[reference] = deepcopy(value)
            return {'signature_ref': reference}
        return {key: visit(item) for key, item in value.items()}

    result = visit(result)
    result['signature_evidence'] = signatures
    result['evidence_meaning'] = (
        'signature_ref resolves to one exact returned declaration in signature_evidence. '
        'Selectors and evidence IDs retain their original provenance. API-use contributions '
        'and input purposes carry their stated author and remain unverified claims; signatures prove only '
        'the declared API shape, not the required task effect.')
    return result
