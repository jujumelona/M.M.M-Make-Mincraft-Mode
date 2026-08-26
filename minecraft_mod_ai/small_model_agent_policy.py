from __future__ import annotations

'Research-derived policy adaptation for the small local planner.\n\nThe policy combines observation-conditioned retry, failure reflection, reusable\nworkflow memory, and task-conditioned strategy composition. Candidate branching stays\nowned by agentic_optimization_contract so search-width policy has one owner.\n'
import hashlib
import json
import os
import stat
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_DISK_LOCK = threading.RLock()
_SESSION: deque[dict[str, Any]] = deque(maxlen=128)
_DISK_CACHE: dict[tuple[Path, int], tuple[tuple[int, int], tuple[dict[str, Any], ...], frozenset[str]]] = {}
_MARKERS = ('networking', 'multiplayer', 'custom_java', 'integration', 'dimension', 'world_event', 'ai_inference', 'agent_tool_use', 'migration', 'persistence', 'dependency', 'asset')
_RECOVERY = {'fixed_point': ('strategy_switch', 'minimal_complete_page', 'avoid_previous_shape'), 'truncated': ('smaller_page', 'complete_records_only', 'cursor_continuation'), 'schema': ('contract_first', 'exact_keys_only', 'type_check_before_finish'), 'no_progress': ('remaining_first', 'new_evidence_required', 'strategy_switch'), 'duplicate': ('catalog_check', 'fresh_identifiers', 'dependency_first'), 'dependency': ('dependency_first', 'existing_exports_only', 'acyclic_order'), 'evidence': ('observable_completion', 'ground_claims_to_ids', 'leave_unsupported_open'), 'other': ('strategy_switch', 'contract_first', 'minimal_complete_page')}

def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, '').strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

def _render(request: Mapping[str, Any] | str) -> str:
    """Render the complete request; paging owns bounded model execution."""
    if isinstance(request, str):
        return request
    try:
        return json.dumps(request, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(request)

def _features(request: Mapping[str, Any] | str, contracts: Sequence[frozenset[str]], stage: str) -> tuple[str, ...]:
    rendered = _render(request)
    lowered = rendered.casefold()
    size = len(rendered.encode('utf-8'))
    result = {'stage:' + stage.casefold().strip(), 'size:' + ('large' if size >= 12 * 1024 else 'medium' if size >= 4 * 1024 else 'small')}
    for contract in contracts:
        result.add('contract:' + ','.join(sorted(str(item) for item in contract)))
    if isinstance(request, Mapping):
        result.update('key:' + str(key) for key in request)
        remaining = request.get('remaining_deliverables')
        if isinstance(remaining, Sequence) and (not isinstance(remaining, (str, bytes))):
            count = len(remaining)
            result.add('remaining:' + ('many' if count >= 6 else 'several' if count >= 3 else 'few'))
    result.update('marker:' + marker for marker in _MARKERS if marker in lowered)
    return tuple(sorted(result))

def _similarity(left: Sequence[str], right: Sequence[str]) -> float:
    a = set(left)
    b = set(right)
    return len(a & b) / max(1, len(a | b)) if a and b else 0.0

def _memory_path() -> Path | None:
    explicit = os.environ.get('MMM_AGENT_WORKFLOW_MEMORY_PATH', '').strip()
    if explicit:
        return Path(explicit).expanduser()
    workspace = os.environ.get('MMM_WORKSPACE', '').strip()
    if not workspace:
        return None
    return Path(workspace).expanduser() / '.minecraft_ai' / 'agent-workflows.jsonl'

def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (int(info.st_mtime_ns), int(info.st_size))

def _disk_snapshot_locked(path: Path, *, limit: int) -> tuple[list[dict[str, Any]], set[str]]:
    """Return recent rows and all known IDs, rescanning only when the file changed."""
    bounded_limit = max(1, int(limit))
    cache_key = (path, bounded_limit)
    fingerprint = _file_fingerprint(path)
    if fingerprint is None:
        _DISK_CACHE.pop(cache_key, None)
        return ([], set())
    cached = _DISK_CACHE.get(cache_key)
    if cached is not None and cached[0] == fingerprint:
        return (list(cached[1]), set(cached[2]))
    rows: deque[dict[str, Any]] = deque(maxlen=bounded_limit)
    identities: set[str] = set()
    try:
        with path.open('r', encoding='utf-8') as handle:
            for raw in handle:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict) or value.get('schema_version') != 'mmm/small-agent-workflow-v1':
                    continue
                identity = str(value.get('workflow_id', '')).strip()
                if identity:
                    identities.add(identity)
                rows.append(value)
    except OSError:
        return ([], set())
    refreshed = _file_fingerprint(path)
    if refreshed is not None:
        _DISK_CACHE[cache_key] = (refreshed, tuple(rows), frozenset(identities))
    return (list(rows), identities)

def _disk_snapshot(path: Path, *, limit: int=128) -> tuple[list[dict[str, Any]], set[str]]:
    with _DISK_LOCK:
        return _disk_snapshot_locked(path, limit=limit)

def _disk_rows(limit: int=128) -> list[dict[str, Any]]:
    path = _memory_path()
    if path is None:
        return []
    rows, _identities = _disk_snapshot(path, limit=limit)
    return rows

def _matches(features: Sequence[str], limit: int=3) -> list[dict[str, Any]]:
    with _LOCK:
        session_rows = list(_SESSION)
    rows = [*session_rows, *_disk_rows()]
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        identity = str(row.get('workflow_id', ''))
        row_features = row.get('features')
        if not identity or identity in seen or (not isinstance(row_features, list)):
            continue
        seen.add(identity)
        score = _similarity(features, [str(item) for item in row_features])
        if score > 0:
            ranked.append((score, identity, row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{'similarity': round(score, 6), 'strategy_modules': list(row.get('strategy_modules', ()))[:8], 'recovered_from': list(row.get('recovered_from', ()))[:4]} for score, _identity, row in ranked[:limit]]

def _strategies(features: Sequence[str], memory: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    chosen = ['contract_first', 'dependency_first', 'observable_completion']
    feature_set = set(features)
    if feature_set & {'marker:networking', 'marker:multiplayer', 'marker:persistence'}:
        chosen.extend(('state_authority_check', 'failure_boundary_check'))
    if feature_set & {'marker:custom_java', 'marker:integration', 'marker:migration'}:
        chosen.extend(('api_contract_check', 'version_constraint_check'))
    if 'size:large' in feature_set or 'remaining:many' in feature_set:
        chosen.extend(('decompose_to_complete_records', 'cursor_continuation'))
    for row in memory[:2]:
        modules = row.get('strategy_modules')
        if isinstance(modules, list):
            chosen.extend(str(item).strip() for item in modules if str(item).strip())
    return tuple(dict.fromkeys(chosen))[:12]

def _failure(exc: BaseException) -> str:
    text = str(exc).casefold()
    if 'fixed point' in text or 'identical invalid' in text or 'repeated identical model output' in text or ('repeated_validation_state' in text) or ('repeated_model_output' in text):
        return 'fixed_point'
    if 'truncat' in text or 'too large' in text or 'overflow' in text:
        return 'truncated'
    if 'no-progress' in text or 'did not advance' in text:
        return 'no_progress'
    if 'duplicate' in text:
        return 'duplicate'
    if 'depend' in text or 'acyclic' in text:
        return 'dependency'
    if 'evidence' in text or 'completion' in text:
        return 'evidence'
    if 'json' in text or 'contract' in text or 'field' in text or ('schema' in text):
        return 'schema'
    return 'other'

def _policy(strategies: Sequence[str], memory: Sequence[Mapping[str, Any]], failure: str, attempt: int) -> str:
    lines = ['HOST SMALL-MODEL PLANNING POLICY:', 'Selected checks: ' + ', '.join(strategies) + '.', 'Validate exact contract, dependencies, and observable completion before finishing.', 'When breadth is too large, emit fewer complete records and continue with the cursor.']
    if memory:
        lines.append(f"A verified workflow with similar structure exists (similarity={memory[0].get('similarity', 0)}); reuse its procedure, not its identifiers.")
    if failure:
        recovery = _RECOVERY.get(failure, _RECOVERY['other'])
        lines.extend((f'Previous attempt failed with observed class={failure}.', 'Change procedure using: ' + ', '.join(recovery) + '.', f'Recovery attempt={attempt}.'))
    return '\n'.join(lines)

def _outcome(page: Mapping[str, Any]) -> dict[str, Any]:

    def count(field: str) -> int:
        value = page.get(field)
        return len(value) if isinstance(value, list) else 0
    return {'fields': sorted(str(key) for key in page), 'modules': count('modules'), 'assets': count('assets'), 'tests': count('acceptance_tests'), 'completed': count('completed_deliverables'), 'complete': page.get('complete') if type(page.get('complete')) is bool else None}

def _record(features: Sequence[str], strategies: Sequence[str], failures: Sequence[str], page: Mapping[str, Any]) -> None:
    row: dict[str, Any] = {'schema_version': 'mmm/small-agent-workflow-v1', 'features': list(features), 'strategy_modules': list(strategies), 'recovered_from': list(dict.fromkeys(failures))[:8], 'outcome': _outcome(page), 'verified_success': True}
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    row['workflow_id'] = 'sha256:' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    with _LOCK:
        if any(item.get('workflow_id') == row['workflow_id'] for item in _SESSION):
            return
        _SESSION.append(row)
    path = _memory_path()
    if path is None:
        return
    with _DISK_LOCK:
        recent_rows, existing = _disk_snapshot_locked(path, limit=128)
        if row['workflow_id'] in existing:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8', newline='\n') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        except OSError:
            return
        recent: deque[dict[str, Any]] = deque(recent_rows, maxlen=128)
        recent.append(row)
        existing.add(str(row['workflow_id']))
        fingerprint = _file_fingerprint(path)
        if fingerprint is not None:
            _DISK_CACHE[path, 128] = (fingerprint, tuple(recent), frozenset(existing))

def planner_search_width_hint(request: Mapping[str, Any] | str, stage: str, *, maximum: int) -> int | None:
    """Estimate useful Best-of-N breadth from prior host-verified workflows.

    Recovery-heavy matches raise breadth; several strong clean matches suppress
    redundant decoding. Ambiguous history returns ``None`` so the deterministic risk
    controller remains authoritative.
    """
    cap = max(1, min(3, int(maximum)))
    if cap <= 1:
        return 1
    matches = _matches(_features(request, (), stage), limit=8)
    strong = [row for row in matches if float(row.get('similarity', 0.0) or 0.0) >= 0.62]
    if not strong:
        return None
    clean_confidence = 0.0
    recovery_pressure = 0.0
    clean_count = 0
    for row in strong:
        similarity = float(row.get('similarity', 0.0) or 0.0)
        recovered = row.get('recovered_from')
        failures = [str(item) for item in recovered] if isinstance(recovered, list) else []
        if failures:
            recovery_pressure += similarity * (1.0 + min(2.0, 0.35 * len(set(failures))))
        else:
            clean_confidence += similarity
            clean_count += 1
    if recovery_pressure >= max(0.9, clean_confidence * 0.45):
        return cap
    if clean_count >= 3 and clean_confidence >= 2.4 and (recovery_pressure <= 0.25):
        return 1
    return None

def enhance_planner(complete_planner_module: Any) -> None:
    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, '_mmm_small_model_agent_policy', False):
        return

    @wraps(current)
    def generate_with_policy(router: Any, *, system_prompt: str, request: dict[str, Any] | str, media_paths: Sequence[Any], expected_contracts: Sequence[frozenset[str]], stage: str) -> dict[str, Any]:
        features = _features(request, expected_contracts, stage)
        memory = _matches(features)
        strategies = _strategies(features, memory)
        production_page = isinstance(request, Mapping) and 'remaining_deliverables' in request
        max_replans = 0 if production_page else _env_int('MMM_SMALL_AGENT_MAX_REPLANS', 2, minimum=0, maximum=3)
        failures: list[str] = []
        for attempt in range(max_replans + 1):
            observed = failures[-1] if failures else ''
            prompt = system_prompt + '\n\n' + _policy(strategies, memory, observed, attempt)
            try:
                page = current(router, system_prompt=prompt, request=request, media_paths=media_paths if attempt == 0 else (), expected_contracts=expected_contracts, stage=stage)
            except complete_planner_module.SpecValidationError as exc:
                failure = _failure(exc)
                if production_page or failure in {'fixed_point', 'no_progress'}:
                    raise
                failures.append(failure)
                strategies = tuple(dict.fromkeys((*strategies, *_RECOVERY.get(failure, _RECOVERY['other']))))[:12]
                if attempt >= max_replans:
                    raise
                continue
            _record(features, strategies, failures, page)
            if failures or memory:
                print('small-model policy:', f'stage={stage}', f'replans={len(failures)}', f'memory={len(memory)}', flush=True)
            return page
        raise complete_planner_module.SpecValidationError(f'{stage} exhausted the bounded small-model policy.')
    generate_with_policy._mmm_small_model_agent_policy = True
    complete_planner_module._generate_json_page_with_repair = generate_with_policy
__all__ = ['enhance_planner', 'planner_search_width_hint']
