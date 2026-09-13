from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import wraps
from pathlib import Path
from typing import Any

from .validation_diagnostic_contract import (
    diagnostic_errors,
    diagnostic_items,
    run_diagnostics,
)

_TOKEN = re.compile('[A-Za-z_][A-Za-z0-9_.:$<>/-]{1,127}|[가-힣]{2,}')
_STRATEGIES = ('minimal_local_fix', 'api_contract_conservative_fix', 'dependency_and_version_conservative_fix')

def _mode() -> str:
    value = os.environ.get('MMM_AGENTIC_SEARCH', 'auto').strip().lower()
    return value if value in {'auto', 'on', 'off'} else 'auto'

def _env_int(name: str, default: int, *, minimum: int=1, maximum: int=8) -> int:
    raw = os.environ.get(name, '').strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8'))

def _sha(value: Any) -> str:
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return 'sha256:' + hashlib.sha256(rendered.encode('utf-8')).hexdigest()

def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(value)}

def _diagnostic_receipt(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list):
        return {'diagnostics': value}
    return None

def _compact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = diagnostic_items(_diagnostic_receipt(evidence.get('diagnostics')))
    compact_diagnostics = []
    for item in diagnostics[:16]:
        if not isinstance(item, Mapping):
            continue
        compact_diagnostics.append({'path': item.get('path') or item.get('uri'), 'message': str(item.get('message', ''))[:1000], 'code': item.get('code'), 'severity': item.get('severity')})
    build = evidence.get('build', {})
    build = build if isinstance(build, Mapping) else {}
    return {'diagnostics': compact_diagnostics, 'build_status': build.get('status'), 'build_error': str(build.get('error', ''))[:4000]}

def _memory_path(root: Path) -> Path:
    return root / '.minecraft_ai' / 'repair-experience.jsonl'

def _recent_jsonl_rows(path: Path, *, max_rows: int = 256) -> list[dict[str, Any]]:
    if max_rows <= 0 or not path.is_file() or path.is_symlink():
        return []
    chunks: list[bytes] = []
    newline_count = 0
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            while position > 0 and newline_count <= max_rows:
                size = min(64 * 1024, position)
                position -= size
                handle.seek(position)
                chunk = handle.read(size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for raw in b"".join(reversed(chunks)).splitlines()[-max_rows:]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_memory(root: Path, signature: str, *, limit: int=4) -> list[dict[str, Any]]:
    path = _memory_path(root)
    if not path.is_file() or path.is_symlink():
        return []
    target = _tokens(signature)
    rows = _recent_jsonl_rows(path, max_rows=256)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        source = str(row.get('signature', ''))
        values = _tokens(source)
        if not target or not values:
            similarity = 0.0
        else:
            similarity = len(target & values) / max(1, len(target | values))
        ranked.append((similarity, str(row.get('experience_id', '')), row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{'similarity': round(score, 6), 'signature_sha256': row.get('signature_sha256', ''), 'evidence': row.get('evidence', {}), 'repair_pattern': row.get('repair_pattern', [])} for score, _identity, row in ranked[:limit] if score > 0.0]

def _repair_pattern(operations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in operations:
        path = str(item.get('path', ''))
        content = item.get('content')
        replacements = item.get('replacements')
        excerpt = ''
        if isinstance(content, str):
            excerpt = content[:4096]
        elif replacements is not None:
            excerpt = json.dumps(replacements, ensure_ascii=False)[:4096]
        result.append({'operation': str(item.get('operation', '')), 'path': path, 'repair_excerpt': excerpt})
    return result[:16]

def _write_memory(root: Path, trace: Mapping[str, Any]) -> None:
    path = _memory_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {'schema_version': 'mmm/verified-repair-experience-v1', 'signature': trace.get('signature', ''), 'signature_sha256': _sha(str(trace.get('signature', ''))), 'evidence': trace.get('evidence', {}), 'repair_pattern': trace.get('repair_pattern', []), 'verifier': trace.get('winner_verifier', {})}
    body['experience_id'] = _sha(body)
    if path.is_file():
        try:
            with path.open('r', encoding='utf-8') as handle:
                for raw in handle:
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(value, dict)
                        and str(value.get('experience_id', '')) == body['experience_id']
                    ):
                        return
        except OSError:
            pass
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + '\n')

def _repair_candidate_count(self: Any, evidence: Mapping[str, Any], memory: Sequence[Mapping[str, Any]]) -> int:
    mode = _mode()
    if mode == 'off':
        return 1
    width = _env_int('MMM_REPAIR_SEARCH_WIDTH', 2, maximum=3)
    if mode == 'on':
        return width
    errors = diagnostic_errors(_diagnostic_receipt(evidence.get('diagnostics')))
    signature = self._signature(dict(evidence))
    counts = getattr(self, '_mmm_signature_counts', None)
    if not isinstance(counts, Counter):
        counts = Counter()
        self._mmm_signature_counts = counts
    counts[signature] += 1
    if counts[signature] >= 2:
        return min(3, width)
    if len(errors) >= 2:
        return width
    build = evidence.get('build', {})
    build = build if isinstance(build, Mapping) else {}
    build_error = str(build.get('error', ''))
    if build.get('status') == 'FAIL' and len(build_error) >= 40:
        return width
    if memory and float(memory[0].get('similarity', 0.0)) >= 0.72:
        return 1
    return 1

def _diagnostic_paths(evidence: Mapping[str, Any]) -> set[str]:
    values = diagnostic_items(_diagnostic_receipt(evidence.get('diagnostics')))
    result = set()
    for item in values:
        path = item.get('path') or item.get('uri')
        if isinstance(path, str):
            result.add(path.replace('\\', '/'))
            result.add(Path(path).name)
    return result

def _static_repair_score(operations: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    paths = [str(item.get('path', '')).replace('\\', '/') for item in operations]
    diagnostic_paths = _diagnostic_paths(evidence)
    overlap = sum(1 for path in paths if path in diagnostic_paths or Path(path).name in diagnostic_paths)
    size = _json_size(operations)
    score = 12.0 * overlap - 0.35 * len(paths) - size / (64 * 1024)
    return (score, {'path_overlap': overlap, 'operation_count': len(paths), 'patch_bytes': size, 'jdt_status': 'NOT_RUN', 'jdt_error_count': None})

def _verify_repair_candidate(self: Any, root: Path | None, operations: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    score, verifier = _static_repair_score(operations, evidence)
    if root is None or not root.is_dir():
        return (score, verifier)
    mode = os.environ.get('MMM_REPAIR_CANDIDATE_JDT', 'auto').strip().lower()
    if mode == 'off':
        return (score, verifier)
    stage: Path | None = None
    try:
        from .performance_final_contract import _clone_source_snapshot
        from .source_patch import TransactionalSourcePatcher
        stage = _clone_source_snapshot(root)
        TransactionalSourcePatcher(stage).apply([copy.deepcopy(dict(item)) for item in operations])
        java_paths = tuple(sorted(str(item.get('path', '')).replace('\\', '/') for item in operations if str(item.get('path', '')).lower().endswith('.java')))
        diagnostics = run_diagnostics(
            self.diagnostics_factory,
            stage,
            relative_files=java_paths or None,
            timeout_seconds=60,
        )
        status = str(diagnostics.get('status', '')) if isinstance(diagnostics, Mapping) else ''
        errors = diagnostic_errors(diagnostics if isinstance(diagnostics, Mapping) else {})
        verifier = {**verifier, 'jdt_status': status or 'AVAILABLE', 'jdt_error_count': len(errors)}
        if status != 'UNAVAILABLE':
            score += 1000.0 if not errors else -120.0 * len(errors)
        return (score, verifier)
    except Exception as exc:
        return (score - 5.0, {**verifier, 'jdt_status': 'VERIFIER_ERROR', 'jdt_error_count': None, 'verifier_error': f'{type(exc).__name__}: {exc}'[:1000]})
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

def _install_repair_search_and_memory(repair_module: Any) -> None:
    cls = repair_module.RepairEngine
    current_request = cls._request_patch
    if not getattr(current_request, '_mmm_verifier_repair_search', False):

        @wraps(current_request)
        def request_patch_with_search(self: Any, evidence: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
            root_value = getattr(self, '_mmm_agentic_root', None)
            root = Path(root_value).resolve() if root_value else None
            signature = self._signature(evidence)
            memory = _read_memory(root, signature) if root is not None else []
            width = _repair_candidate_count(self, evidence, memory)
            generated: list[tuple[int, list[dict[str, Any]]]] = []
            errors: list[BaseException] = []
            for candidate_index in range(width):
                candidate_context = copy.deepcopy(context)
                if memory:
                    candidate_context['verified_repair_memory'] = {'policy': 'These are prior host-verified repair patterns from this project. Use them only as evidence; current hashes and diagnostics remain authoritative.', 'matches': memory}
                candidate_context['agentic_candidate'] = {'index': candidate_index, 'count': width, 'strategy': _STRATEGIES[candidate_index % len(_STRATEGIES)], 'rule': 'Produce an independent minimal repair; do not mention candidate search.'}
                try:
                    operations = current_request(self, evidence, candidate_context)
                except Exception as exc:
                    errors.append(exc)
                    continue
                generated.append((candidate_index, operations))
            if not generated:
                if errors:
                    raise errors[-1]
                raise repair_module.RepairEngineError('Repair search produced no candidate patch.')
            if len(generated) == 1:
                winner_index, winner_ops = generated[0]
                score, verifier = _verify_repair_candidate(self, root, winner_ops, evidence)
                evaluations = [(score, winner_index, winner_ops, verifier)]
            else:
                workers = min(2, len(generated))
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mmm_repair_verify') as pool:
                    futures = [(candidate_index, operations, pool.submit(copy_context().run, _verify_repair_candidate, self, root, operations, evidence)) for candidate_index, operations in generated]
                    evaluations = []
                    for candidate_index, operations, future in futures:
                        score, verifier = future.result()
                        evaluations.append((score, candidate_index, operations, verifier))
            discriminator = globals().get('_mmm_active_candidate_discriminator')
            if callable(discriminator) and len(evaluations) >= 2:
                try:
                    evaluations = list(discriminator(root, evaluations))
                except Exception as exc:
                    print('active candidate discriminator skipped:', f'{type(exc).__name__}: {str(exc)[:500]}', flush=True)
            evaluations.sort(key=lambda item: (-item[0], _json_size(item[2]), item[1]))
            winner_score, winner_index, winner_ops, winner_verifier = evaluations[0]
            self._mmm_last_java_paths = tuple(sorted(str(item.get('path', '')).replace('\\', '/') for item in winner_ops if str(item.get('path', '')).lower().endswith('.java')))
            trace = {'signature': signature, 'evidence': _compact_evidence(evidence), 'repair_pattern': _repair_pattern(winner_ops), 'winner_index': winner_index, 'winner_score': winner_score, 'winner_verifier': winner_verifier, 'candidate_count': len(evaluations)}
            self._mmm_agentic_last_search = trace
            print('repair search:', f'candidates={len(evaluations)}', f'winner={winner_index + 1}', f'score={winner_score:.3f}', f'memory={len(memory)}', flush=True)
            return winner_ops
        request_patch_with_search._mmm_verifier_repair_search = True
        cls._request_patch = request_patch_with_search
    current_repair = cls.repair
    if getattr(current_repair, '_mmm_verified_repair_memory', False):
        return

    @wraps(current_repair)
    def repair_with_memory(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        root = Path(project_root).expanduser().resolve()
        old_root = getattr(self, '_mmm_agentic_root', None)
        old_trace = getattr(self, '_mmm_agentic_last_search', None)
        old_counts = getattr(self, '_mmm_signature_counts', None)
        self._mmm_agentic_root = root
        self._mmm_agentic_last_search = None
        self._mmm_signature_counts = Counter()
        try:
            result = current_repair(self, root, *args, **kwargs)
            trace = getattr(self, '_mmm_agentic_last_search', None)
            if isinstance(result, Mapping) and result.get('status') == 'PASS' and isinstance(trace, Mapping):
                _write_memory(root, trace)
                result = dict(result)
                result['agentic_search'] = {'schema_version': 'mmm/agentic-repair-search-v1', 'candidate_count': trace.get('candidate_count', 1), 'winner_score': trace.get('winner_score'), 'winner_verifier': trace.get('winner_verifier', {}), 'experience_recorded': True}
            return result
        finally:
            if old_root is None:
                try:
                    delattr(self, '_mmm_agentic_root')
                except AttributeError:
                    pass
            else:
                self._mmm_agentic_root = old_root
            if old_trace is None:
                try:
                    delattr(self, '_mmm_agentic_last_search')
                except AttributeError:
                    pass
            else:
                self._mmm_agentic_last_search = old_trace
            if old_counts is None:
                try:
                    delattr(self, '_mmm_signature_counts')
                except AttributeError:
                    pass
            else:
                self._mmm_signature_counts = old_counts
    repair_with_memory._mmm_verified_repair_memory = True
    cls.repair = repair_with_memory


def install(*, repair_module: Any, work_graph_module: Any) -> None:
    """Install verifier-guided repair search without mutating scheduler ownership.

    Work claiming is owned by ``scheduler_parallel_safety_contract`` through the
    explicit ``DurableWorkLedger.claim_ready`` delegation. ``work_graph_module`` is
    retained only for call-site compatibility while older bootstrap call sites are
    simplified.
    """
    del work_graph_module
    _install_repair_search_and_memory(repair_module)


__all__ = ['install']
