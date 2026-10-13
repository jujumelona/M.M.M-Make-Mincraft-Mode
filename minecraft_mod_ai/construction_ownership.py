"""Exactly one behavior owner for each residual runtime record."""
from copy import deepcopy
import json
from .request_runtime_records import owner_records
from .fixed_template_generation import _validate_native_arguments


def assign_record_owners(plan, behaviors, router):
    saved = plan.content_design.get('_construction_record_owners', {})
    assignments, expected = {}, {}
    for parent in plan.request_scope.get('requirements', ()):
        rid = parent['requirement_id']
        candidates = [b for b in behaviors if b.requirement_id == rid]
        for record in owner_records(plan.structured_sections, plan.content_design.get('_request_runtime_owners', ()), rid):
            ref = record['record_ref']
            expected[ref] = {b.behavior_id for b in candidates}
            if ref in saved:
                assignments[ref] = saved[ref]
                continue
            if len(candidates) == 1:
                assignments[ref] = candidates[0].behavior_id
                continue
            if router is None:
                raise ValueError('CONSTRUCTION_RECORD_OWNER_REQUIRED: ' + ref)
            matches = []
            for candidate in candidates:
                schema = {'type': 'object', 'properties': {'owns': {'type': 'boolean'}}, 'required': ['owns'], 'additionalProperties': False}
                payload = {'record': record, 'behavior': candidate.to_dict()}
                result = _validate_native_arguments(router.generate_structured_data('planner', [
                    {'role': 'system', 'content': 'Does this behavior own implementation of this exact record? Preserve its action identity. Each record must have exactly one behavior owner.'},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}], data_name='construction_record_owner', parameters=schema), schema)
                if result['owns']:
                    matches.append(candidate.behavior_id)
            if len(matches) != 1:
                raise ValueError('CONSTRUCTION_RECORD_OWNER_AMBIGUOUS: ' + ref)
            assignments[ref] = matches[0]
    if set(saved) - set(expected) or any(owner not in expected[ref] for ref, owner in assignments.items()):
        raise ValueError('CONSTRUCTION_RECORD_OWNER_INVALID')
    return assignments


def scoped_state(section, reads, writes):
    """Select complete state declarations/constraints referencing current state IDs."""
    names = set(reads) | set(writes)
    def references(value):
        if isinstance(value, str):
            import re
            return bool(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', value)) & names)
        if isinstance(value, dict):
            return any(references(v) for v in value.values())
        if isinstance(value, list):
            return any(references(v) for v in value)
        return False
    specification = section.get('specification', {})
    return {'specification': {key: [deepcopy(row) for row in rows if references(row)]
                             for key, rows in specification.items() if isinstance(rows, list)}}


def assign_behavior_records(router, behaviors, records):
    """Assign each residual runtime record to exactly one observable behavior.

    Typed/UI records are already removed by the parent ownership contract.
    No transaction or state mutation is duplicated across child callbacks.
    """
    from .construction_author import ask, obj
    by_id = {row['behavior_id']: row for row in behaviors}
    if not by_id or len(by_id) != len(behaviors):
        raise ValueError('CONSTRUCTION_BEHAVIOR_OWNER_IDENTITY_INVALID')
    assignments = {}
    for record in records:
        ref = record['record_ref']
        if ref in assignments:
            raise ValueError('CONSTRUCTION_RECORD_DUPLICATE: ' + ref)
        if len(by_id) == 1:
            assignments[ref] = next(iter(by_id))
            continue
        schema = obj({'behavior_id': {'type': 'string', 'enum': sorted(by_id)}})
        payload = {'record_ref': ref, 'record': record['record'],
                   'behaviors': [{'behavior_id': row['behavior_id'],
                                 'intent': row['intent'], 'effects': row['effects']}
                                for row in behaviors]}
        reply = ask(router, 'construction_record_owner', payload, schema,
                    'Select exactly one child behavior to implement this entire '
                    'runtime record and its state mutations. Do not invent a new '
                    'behavior or assign a record to multiple callbacks.')
        assignments[ref] = reply['behavior_id']
    return assignments
