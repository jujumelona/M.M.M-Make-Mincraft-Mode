"""Current-source prompts and measured-context recovery for Java file authors."""
from __future__ import annotations

import json
import re
from copy import deepcopy

from .construction_type_contracts import canonical_type_name


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def file_messages(system, prompt, repair=None):
    """Keep the task and latest complete failed file, without replay history."""
    current = deepcopy(prompt)
    if repair:
        repair = deepcopy(repair)
        diagnostics = repair.get('verified_symbols_from_compiler_failure', ())
        api = current.setdefault('inspected_target_api', {})
        if isinstance(api, dict):
            # Retain verified contracts in the durable task view. Replacing
            # the latest failed-source message must not erase discovery or
            # contracts learned from an earlier compiler diagnostic.
            records = api.setdefault('class_members', [])
            by_owner = {row.get('verified_class'): row for row in records}
            for latest in diagnostics:
                initial = by_owner.get(latest.get('verified_class'))
                if initial is None:
                    records.append(latest)
                    by_owner[latest.get('verified_class')] = latest
                    continue
                combined = {_json(row): row for row in latest.get('members', ())}
                combined.update({_json(row): row for row in initial.get('members', ())})
                initial['members'] = list(combined.values())
                initial['declaration'] = {**initial.get('declaration', {}), **latest.get('declaration', {})}
                counts = dict(initial.get('overload_counts', {}))
                counts.update(latest.get('overload_counts', {}))
                if counts:
                    initial['overload_counts'] = counts
                initial['member_count'] = max(initial.get('member_count', 0),
                    latest.get('member_count', 0), len(combined))
                initial['members_truncated'] = len(combined) < initial['member_count']
            if diagnostics:
                repair['verified_symbol_owners_from_compiler_failure'] = [
                    row.get('verified_class') for row in diagnostics]
                repair['verified_symbols_from_compiler_failure'] = []
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': _json(current)}]
    if repair:
        messages.append({'role': 'user', 'content': _json(repair)})
    return messages


def _reduce_records(records, required_names):
    answer, changed = [], False
    for record in records:
        current = deepcopy(record)
        members = current.get('members', ())
        optional = []
        for index, row in enumerate(members):
            result_owner = canonical_type_name(str(row.get('return_type', '')).split('<', 1)[0])
            factory = (row.get('kind') == 'CONSTRUCTOR' or
                       row.get('static') and result_owner == record.get('verified_class'))
            if not factory and row.get('name') not in required_names | {'of', 'build', 'register'}:
                optional.append(index)
        # Keep all referenced overloads/factories. Only supplemental discovery
        # observations are reduced; a type or source string is never sliced.
        removed = set(optional[len(optional) // 2:])
        if removed:
            current['members'] = [row for index, row in enumerate(members) if index not in removed]
            current['members_truncated'] = True
            shown = {row.get('name') for row in current['members']}
            if 'overload_counts' in current:
                current['overload_counts'] = {name: count for name, count in current['overload_counts'].items()
                                             if name in shown}
            changed = True
        answer.append(current)
    return answer, changed


def compact_api_observations(messages):
    """Reduce only optional API rows; preserve every mandatory input verbatim."""
    parsed, required = {}, set()
    api_fields = {'inspected_target_api', 'verified_symbols_from_compiler_failure'}
    for index, message in enumerate(messages):
        if message.get('role') != 'user' or not isinstance(message.get('content'), str):
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        parsed[index] = value
        mandatory = {key: item for key, item in value.items() if key not in api_fields}
        required.update(re.findall(r'\b[A-Za-z_$][\w$]*', _json(mandatory)))
    result, changed = deepcopy(messages), False
    for index, value in parsed.items():
        api = value.get('inspected_target_api')
        if isinstance(api, dict):
            reduced, progress = _reduce_records(api.get('class_members', ()), required)
            if progress:
                api['class_members'] = reduced
                changed = True
        if isinstance(value.get('verified_symbols_from_compiler_failure'), list):
            reduced, progress = _reduce_records(value['verified_symbols_from_compiler_failure'], required)
            if progress:
                value['verified_symbols_from_compiler_failure'] = reduced
                changed = True
        result[index]['content'] = _json(value)
    if changed and len(_json(result).encode()) < len(_json(messages).encode()):
        return result
    return None


def _record_context_pressure(messages, candidate, boundary, target_path):
    from .root_cause_trace import durable_trace_path, emit_root_cause
    from .planner_trace_artifacts import save_trace_artifact

    details = {'path': target_path, 'input_tokens': boundary.prompt_tokens,
               'before_bytes': len(_json(messages).encode()),
               'after_bytes': len(_json(candidate).encode()) if candidate is not None else None,
               'recovered_observations': candidate is not None}
    try:
        details['prompt_artifact'] = save_trace_artifact(
            {'messages': messages, **details}, durable_trace_path().parent / 'file_author_context')
    except (OSError, ValueError) as exc:
        details['prompt_artifact_error'] = type(exc).__name__
    emit_root_cause('construction_file_context_compaction', stage='generate',
                   operation=target_path, result='RETRY' if candidate is not None else 'BLOCKED',
                   details=details)


def generate_file_text(router, messages, *, target_path):
    """Context preflight rejections do not spend the compiler repair contract."""
    from .llama_finish_reason_contract import (
        CONTEXT_PRESSURE, completion_boundary_error, completion_boundary_kind)

    while True:
        try:
            return router.generate_text('coder', messages, enable_tools=False, force_non_thinking=True)
        except Exception as exc:
            boundary = completion_boundary_error(exc)
            if boundary is None or completion_boundary_kind(exc) != CONTEXT_PRESSURE:
                raise
            candidate = compact_api_observations(messages)
            _record_context_pressure(messages, candidate, boundary, target_path)
            if candidate is None:
                # Required task/source/signatures still do not fit. Keep the
                # original typed error; never retry an unchanged payload.
                raise
            messages[:] = candidate
