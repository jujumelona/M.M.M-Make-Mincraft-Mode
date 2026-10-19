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
        # A repair is about the current complete source and actual diagnostics.
        # Imported-but-unused classes and dependencies of unused API methods do
        # not justify growing the next prompt into a server/classpath catalog.
        from .construction_file_author import _lexical_code
        source = _lexical_code(str(repair.get('previous_source') or ''))
        static_imports = re.findall(r'\bimport\s+static\s+([\w.$]+)\.(\*|[\w$]+)\s*;', source)
        source = re.sub(r'\bimport\s+(?:static\s+)?[\w.*$]+\s*;', '', source)
        referenced = set(re.findall(r'\b[A-Za-z_$][\w$]*',
                                   source + '\n' + str(repair.get('error') or '')))
        api = current.setdefault('inspected_target_api', {})
        if isinstance(api, dict):
            # Retain verified contracts in the durable task view. Replacing
            # the latest failed-source message must not erase discovery or
            # contracts learned from an earlier compiler diagnostic.
            records = api.setdefault('class_members', [])
            for record in records:
                record.pop('unavailable_requested_members', None)
            by_owner = {row.get('verified_class'): row for row in records}
            for latest in diagnostics:
                owner = str(latest.get('verified_class') or '')
                initial = by_owner.get(owner)
                if initial is None:
                    static_reference = any(
                        scope.replace('$', '.') == owner and
                        (member in referenced or member == '*' and any(
                            row.get('static') and row.get('name') in referenced
                            for row in latest.get('members', ())))
                        for scope, member in static_imports)
                    if owner.rsplit('.', 1)[-1] not in referenced and not static_reference:
                        continue
                    records.append(latest)
                    by_owner[owner] = latest
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
                if latest.get('unavailable_requested_members'):
                    initial['unavailable_requested_members'] = latest['unavailable_requested_members']
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


def _record_context_pressure(messages, candidate, boundary, target_path, *, kind):
    from .planner_trace_artifacts import save_trace_artifact
    from .root_cause_trace import durable_trace_path, emit_root_cause

    details = {'path': target_path, 'input_tokens': boundary.prompt_tokens if boundary is not None else 0,
               'boundary_kind': kind,
               'before_bytes': len(_json(messages).encode()),
               'after_bytes': len(_json(candidate).encode()) if candidate is not None else None,
               'recovered_observations': candidate is not None}
    try:
        details['prompt_artifact'] = save_trace_artifact(
            {'messages': messages, **details}, durable_trace_path().parent / 'file_author_context')
    except (OSError, ValueError) as exc:
        details['prompt_artifact_error'] = type(exc).__name__
    event = {
        'context_pressure': 'construction_file_context_compaction',
        'output_exhausted': 'construction_file_output_compaction',
        'repeated_output': 'construction_file_repetition_compaction',
    }[kind]
    emit_root_cause(event, stage='generate',
                   operation=target_path, result='RETRY' if candidate is not None else 'BLOCKED',
                   details=details)


def _degenerate_decode_error(exc):
    """Detect the actual SSE watchdog failure even behind a model adapter wrapper."""
    from .llama_completion_liveness_contract import LlamaDegenerateOutputError
    current, visited = exc, set()
    while isinstance(current, BaseException) and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, LlamaDegenerateOutputError):
            return current
        current = (getattr(current, 'cause', None)
                   or current.__cause__ or current.__context__)
    return None


def _completion_retry_candidate(messages, exc, target_path, *, output_retry_used,
                                repetition_retry_used=False):
    """Never continue partial Java; recover stalled decoding by changing the source protocol."""
    from .llama_finish_reason_contract import (
        CONTEXT_PRESSURE,
        OUTPUT_EXHAUSTED,
        completion_boundary_error,
        completion_boundary_kind,
    )

    boundary = completion_boundary_error(exc)
    kind = completion_boundary_kind(exc)
    if boundary is None or kind not in {CONTEXT_PRESSURE, OUTPUT_EXHAUSTED}:
        if repetition_retry_used or _degenerate_decode_error(exc) is None:
            return None
        # A repeated import/identifier stream is a decoder mode collapse,
        # not a context shortage. Merely removing a few optional API records
        # reproduced the failure in production. Switch the Java *protocol*:
        # fully-qualified references, no imports, no import-list prefix.
        candidate = compact_api_observations(messages) or deepcopy(messages)
        for message in candidate:
            if message.get('role') != 'system':
                continue
            original = str(message.get('content') or '')
            message['content'] = original.replace(
                'include all necessary imports,',
                'do not emit import declarations (use fully qualified Java names),',
            ).replace(
                'include only imports for types actually referenced by the completed class.',
                'do not emit import declarations (use fully qualified Java names).',
            ) + (
                "\nREPETITION RECOVERY: The previous decoder became stuck emitting "
                "repeated Java imports. After the package statement, start the "
                "public class declaration immediately. Emit ZERO import lines. "
                "Use fully qualified type names in the implementation. "
                "Write the complete real feature; no placeholders, no skipped "
                "behavior, and no commentary. Never repeat an import or type list."
            )
        _record_context_pressure(
            messages, candidate, boundary, target_path, kind='repeated_output')
        return candidate, 'repeated_output'
    if kind == OUTPUT_EXHAUSTED and output_retry_used:
        return None
    candidate = compact_api_observations(messages)
    _record_context_pressure(messages, candidate, boundary, target_path, kind=kind)
    if candidate is None:
        return None
    return candidate, kind


def generate_file_text(router, messages, *, target_path):
    """Reduce optional API evidence, never truncate or checkpoint partial Java."""
    output_retry_used = False
    repetition_retry_used = False
    while True:
        try:
            return router.generate_text('coder', messages, enable_tools=False, force_non_thinking=True)
        except Exception as exc:
            retry = _completion_retry_candidate(
                messages, exc, target_path, output_retry_used=output_retry_used,
                repetition_retry_used=repetition_retry_used)
            if retry is None:
                raise
            candidate, kind = retry
            output_retry_used |= kind == 'output_exhausted'
            repetition_retry_used |= kind == 'repeated_output'
            messages[:] = candidate
