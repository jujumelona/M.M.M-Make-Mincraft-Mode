"""Carry canonical artifact responsibility into bounded construction requests."""
from __future__ import annotations

import json


_SCOPED_BINDINGS = frozenset({
    'construction_task_json', 'construction_tasks_json', 'construction_behaviors_json',
    'owned_runtime_records_json', 'retained_typed_behavior_json',
    'verified_source_code_json', 'target_contract_json',
})


def construction_contract(manifest, inputs, spec):
    bindings = spec['bindings']
    pages = {}
    scalar = {}
    for key, value in bindings.items():
        if key in _SCOPED_BINDINGS or key in {'source_requirement', 'java_implementation_requirements'}:
            continue
        if key.endswith('_json'):
            decoded = json.loads(value)
            if isinstance(decoded, list): units = decoded
            elif isinstance(decoded, dict): units = [{'key': k, 'value': v} for k, v in decoded.items()]
            else: units = [decoded]
            pages[key] = units
        else:
            scalar[key] = value
    contract = {
        'canonical_leaf': manifest['id'], 'responsibility': manifest['task'],
        'rules': list(manifest.get('rules', ())),
        'host_requirements': manifest.get('host_requirements', {}),
        'identifiers': {port['name']: inputs[port['name']] for port in manifest['inputs']
                        if port['type'] == 'identifier' and port['name'] in inputs},
        'implementation_requirements': bindings.get('java_implementation_requirements', ''),
        'bindings': scalar,
        'binding_pages': [{'binding_id': key, 'unit_count': len(pages[key])} for key in sorted(pages)],
    }
    return contract, pages
