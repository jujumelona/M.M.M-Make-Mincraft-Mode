from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any

from .custom_generation_research import (
    _fork_router_for_candidate,
    _public_signature_without_target_defaults,
    _ResearchEvidenceRouter,
    _run_single_with_research,
    _StrategyRouter,
    _target_values,
)

_STRATEGIES = (
    "minimal_surface_area",
    "api_contract_first",
    "runtime_and_persistence_first",
)

def _mode() -> str:
    value = os.environ.get('MMM_AGENTIC_SEARCH', 'auto').strip().lower()
    return value if value in {'auto', 'on', 'off'} else 'auto'


def _active_native_slots() -> int:
    raw = os.environ.get('MMM_LLAMA_ACTIVE_PARALLEL', '1').strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _width(module: Any) -> int:
    mode = _mode()
    if mode == 'off':
        return 1
    raw = os.environ.get('MMM_CUSTOM_SEARCH_WIDTH', '2').strip()
    try:
        configured = int(raw)
    except ValueError:
        configured = 2
    configured = max(1, min(3, configured))
    if mode == 'on':
        return configured
    slots = _active_native_slots()
    if slots <= 1:
        return 1
    kind = str(getattr(module, 'kind', ''))
    config = getattr(module, 'config', {})
    config = config if isinstance(config, Mapping) else {}
    depends = tuple(getattr(module, 'depends_on', ()) or ())
    gates = tuple(getattr(module, 'required_gates', ()) or ())
    risk = int(kind in {'custom_java', 'integration', 'structure', 'biome', 'dimension', 'world_event'})
    rendered = json.dumps(config, ensure_ascii=False, sort_keys=True)
    if len(rendered.encode('utf-8')) >= 2048 or len(depends) >= 2 or len(gates) >= 2:
        risk += 1
    return min(configured, slots) if risk >= 2 else 1


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8'))


def _sha256_receipt(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _candidate_patch_capture(*, base_root: Path, candidate_root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    receipt = result.get('patch_receipt')
    receipt_ops = receipt.get('operations') if isinstance(receipt, Mapping) else None
    if not isinstance(receipt_ops, list) or not receipt_ops:
        raise RuntimeError('Custom candidate has no staged patch receipt.')

    operations: list[dict[str, Any]] = []
    before: dict[str, bytes | None] = {}
    seen_paths: set[str] = set()
    for item in receipt_ops:
        if not isinstance(item, Mapping):
            raise RuntimeError('Custom candidate patch receipt is malformed.')
        relative = str(item.get('path', '')).strip()
        operation = str(item.get('operation', '')).strip().lower()
        if not relative:
            raise RuntimeError('Custom candidate patch receipt has an empty path.')
        if relative in seen_paths:
            raise RuntimeError(f'Custom candidate patch receipt repeats path: {relative}')
        seen_paths.add(relative)
        if operation not in {'create', 'replace', 'edit', 'delete'}:
            raise RuntimeError(
                f'Custom candidate patch receipt has invalid operation for {relative}: {operation!r}'
            )

        base_path = (base_root / relative).resolve()
        candidate_path = (candidate_root / relative).resolve()
        try:
            base_path.relative_to(base_root)
            candidate_path.relative_to(candidate_root)
        except ValueError as exc:
            raise RuntimeError(f'Custom candidate path escaped staging root: {relative}') from exc
        if base_path.is_symlink() or candidate_path.is_symlink():
            raise RuntimeError(f'Custom candidate patch path may not be a symlink: {relative}')

        base_bytes = base_path.read_bytes() if base_path.is_file() else None
        candidate_bytes = candidate_path.read_bytes() if candidate_path.is_file() else None
        before[relative] = base_bytes
        before_sha = item.get('before_sha256')
        after_sha = item.get('after_sha256')

        if operation == 'create':
            if before_sha is not None or base_bytes is not None:
                raise RuntimeError(f'Custom candidate create precondition is invalid: {relative}')
            if candidate_bytes is None:
                raise RuntimeError(f'Custom candidate create output is missing: {relative}')
            actual_after = _sha256_receipt(candidate_bytes)
            if str(after_sha) != actual_after:
                raise RuntimeError(
                    f'Custom candidate after hash drifted for {relative}: {actual_after} != {after_sha}'
                )
            operations.append(
                {'operation': 'create', 'path': relative, 'content': candidate_bytes.decode('utf-8')}
            )
            continue

        if base_bytes is None:
            raise RuntimeError(f'Custom candidate base file is missing: {relative}')
        actual_before = _sha256_receipt(base_bytes)
        if str(before_sha) != actual_before:
            raise RuntimeError(
                f'Custom candidate base hash drifted for {relative}: {actual_before} != {before_sha}'
            )

        if operation == 'delete':
            if after_sha is not None:
                raise RuntimeError(f'Custom candidate delete has an after hash: {relative}')
            if candidate_path.exists() or candidate_bytes is not None:
                raise RuntimeError(f'Custom candidate delete output still exists: {relative}')
            operations.append(
                {'operation': 'delete', 'path': relative, 'expected_sha256': actual_before}
            )
            continue

        if candidate_bytes is None:
            raise RuntimeError(f'Custom candidate replacement output is missing: {relative}')
        actual_after = _sha256_receipt(candidate_bytes)
        if str(after_sha) != actual_after:
            raise RuntimeError(
                f'Custom candidate after hash drifted for {relative}: {actual_after} != {after_sha}'
            )
        operations.append(
            {
                'operation': 'replace',
                'path': relative,
                'expected_sha256': actual_before,
                'content': candidate_bytes.decode('utf-8'),
            }
        )
    return {'operations': operations, 'before': before}


def _clone_candidate_snapshot(base_root: Path, *, candidate_index: int, performance_module: Any) -> Path:
    workspace = Path(
        tempfile.mkdtemp(prefix=f'candidate-{candidate_index:02d}-', dir=base_root.parent)
    ).resolve()
    candidate_root = workspace / 'project'
    shutil.copytree(
        base_root,
        candidate_root,
        copy_function=performance_module._reflink_or_copy,
    )
    return candidate_root


def _capture_candidate(
    owner: Any,
    single_generate: Any,
    candidate_root: Path,
    *,
    strategy: str,
    candidate_index: int,
    count: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one isolated candidate and return its verified rebase capture.

    ``coder_max_efficiency_contract`` owns the shared-base fan-out, while this
    module owns candidate research/strategy routing and the exact before/after
    patch contract.  Keep an immutable, copy-on-write sibling snapshot until the
    generated receipt has been checked so a later concurrent winner commit still
    has the base bytes required for a three-way rebase.
    """

    from . import performance_final_contract as performance_module

    resolved_root = Path(candidate_root).expanduser().resolve()
    if not resolved_root.is_dir() or resolved_root.is_symlink():
        raise RuntimeError("Custom candidate root must be a regular directory.")
    base_root = performance_module._clone_snapshot_tree(
        resolved_root,
        parent=resolved_root.parent,
        prefix=f"candidate-base-{candidate_index:02d}-",
    )
    previous_router = owner.router
    result: dict[str, Any] | None = None
    checkpoint_handed_off = False
    owner.router = _StrategyRouter(
        previous_router,
        strategy=strategy,
        candidate_index=candidate_index,
        count=count,
    )
    try:
        result = _run_single_with_research(
            owner,
            single_generate,
            resolved_root,
            args=args,
            kwargs=kwargs,
        )
        capture = _candidate_patch_capture(
            base_root=base_root,
            candidate_root=resolved_root,
            result=result,
        )
        checkpoint_handed_off = True
        return result, capture
    finally:
        if result is not None and not checkpoint_handed_off:
            owner.release_generation_checkpoint(result)
        owner.router = previous_router
        shutil.rmtree(base_root, ignore_errors=True)


def _verify_candidate(candidate_root: Path, result: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    touched = [str(value).replace('\\', '/') for value in result.get('touched_paths', []) if isinstance(value, str)]
    java_paths = tuple(sorted(path for path in touched if path.lower().endswith('.java')))
    operation_count = int(result.get('operation_count', 0) or 0)
    runtime_tests = result.get('runtime_tests', [])
    runtime_tests = runtime_tests if isinstance(runtime_tests, list) else []
    research = result.get('research_code_context')
    research_score = min(2.0, float(research.get('evidence_count', 0)) / 4.0) if isinstance(research, Mapping) else 0.0
    score = 2.0 * len(runtime_tests) - 0.3 * operation_count - 0.05 * len(touched) + research_score
    verifier: dict[str, Any] = {'operation_count': operation_count, 'touched_path_count': len(touched), 'runtime_test_count': len(runtime_tests), 'research_evidence_score': research_score, 'jdt_status': 'NOT_RUN', 'jdt_error_count': None}
    if not java_paths or os.environ.get('MMM_CUSTOM_CANDIDATE_JDT', 'auto').strip().lower() == 'off':
        return (score, verifier)
    try:
        from .java_lsp import JavaLanguageService
        from .repair_diagnostics_contract import diagnostic_errors
        diagnostics = JavaLanguageService().diagnostics(candidate_root, relative_files=java_paths, timeout_seconds=60)
        errors = diagnostic_errors(diagnostics)
        verifier['jdt_status'] = 'AVAILABLE'
        verifier['jdt_error_count'] = len(errors)
        score += 1000.0 if not errors else -120.0 * len(errors)
    except Exception as exc:
        verifier['jdt_status'] = 'VERIFIER_ERROR'
        verifier['verifier_error'] = f'{type(exc).__name__}: {exc}'[:1000]
        score -= 5.0
    return (score, verifier)


def install(custom_module_generator_module: Any) -> None:
    from . import performance_final_contract as performance_module
    from . import source_patch as source_patch_module

    performance_module._install_locked_source_patcher(source_patch_module)
    cls = custom_module_generator_module.CustomModuleGenerator
    original = cls.generate
    if getattr(original, '_mmm_research_generation_search', False):
        return

    @wraps(original)
    def generate_with_search(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        module = kwargs.get('module')
        count = _width(module)
        if count <= 1:
            return _run_single_with_research(
                self,
                original,
                project_root,
                args=args,
                kwargs=kwargs,
            )

        live_root = Path(project_root).expanduser().resolve()
        if not live_root.is_dir() or live_root.is_symlink():
            return _run_single_with_research(
                self,
                original,
                project_root,
                args=args,
                kwargs=kwargs,
            )

        base_root = performance_module._clone_source_snapshot(live_root)
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        errors: dict[int, BaseException] = {}
        winner_index: int | None = None
        winner_checkpoint_acknowledged = False
        winner_live_committed = False

        def solve(candidate_index: int) -> tuple[int, Path, dict[str, Any]]:
            candidate_root = _clone_candidate_snapshot(
                base_root,
                candidate_index=candidate_index,
                performance_module=performance_module,
            )
            worker = copy.copy(self)
            worker._cached_index = None
            worker._cached_root = None
            strategy = _STRATEGIES[candidate_index % len(_STRATEGIES)]
            worker.router = _StrategyRouter(
                _fork_router_for_candidate(self.router),
                strategy=strategy,
                candidate_index=candidate_index,
                count=count,
            )
            try:
                result = _run_single_with_research(
                    worker,
                    original,
                    candidate_root,
                    args=args,
                    kwargs=kwargs,
                )
                if not isinstance(result, dict):
                    raise RuntimeError('Custom generation candidate returned a non-object receipt.')
                return candidate_index, candidate_root, result
            except BaseException:
                shutil.rmtree(candidate_root.parent, ignore_errors=True)
                raise

        try:
            workers = min(count, _active_native_slots())
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix='mmm_custom_generate',
            ) as pool:
                futures = [pool.submit(solve, index) for index in range(count)]
                for candidate_index, future in enumerate(futures):
                    try:
                        candidates.append(future.result())
                    except BaseException as exc:
                        errors[candidate_index] = exc
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise

            candidates.sort(key=lambda item: item[0])
            if not candidates:
                if errors:
                    raise errors[max(errors)]
                raise RuntimeError('Custom generation search produced no candidate.')

            def verify(item: tuple[int, Path, dict[str, Any]]):
                index, candidate_root, result = item
                score, verifier = _verify_candidate(candidate_root, result)
                return score, index, candidate_root, result, verifier

            if len(candidates) == 1:
                evaluations = [verify(candidates[0])]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(2, len(candidates)),
                    thread_name_prefix='mmm_custom_verify',
                ) as pool:
                    evaluations = list(pool.map(verify, candidates))

            evaluations.sort(
                key=lambda item: (
                    -float(item[0]),
                    _json_size(item[3]),
                    int(item[1]),
                )
            )
            score, winner_index, winner_root, result, verifier = evaluations[0]
            capture = _candidate_patch_capture(
                base_root=base_root,
                candidate_root=winner_root,
                result=result,
            )
            commit_receipt = performance_module._commit_staged_operations(
                live_root=live_root,
                staging_root=winner_root,
                capture=capture,
                source_patch_module=source_patch_module,
            )
            winner_live_committed = True
            rewritten = performance_module._rewrite_root_paths(
                result,
                winner_root,
                live_root,
            )
            rewritten['patch_receipt'] = commit_receipt
            rewritten['agentic_generation_search'] = {
                'schema_version': 'mmm/custom-generation-search-v3',
                'candidate_count': len(evaluations),
                'candidate_workers': workers,
                'winner_index': int(winner_index),
                'winner_score': float(score),
                'winner_verifier': verifier,
                'candidate_scores': [
                    {
                        'candidate_index': int(item[1]),
                        'score': float(item[0]),
                        'verifier': item[4],
                    }
                    for item in sorted(evaluations, key=lambda value: value[1])
                ],
                'research_aware': True,
                'dependency_admission': 'exact',
            }
            # The base generator's checkpoint is intentionally retained while this
            # candidate is only staged.  Acknowledge its opaque cleanup token only
            # after the winning patch has committed to the live project above.
            winner_checkpoint_acknowledged = bool(
                self.acknowledge_generation_checkpoint(rewritten)
            )
            rewritten['generation_checkpoint_acknowledged'] = (
                winner_checkpoint_acknowledged
            )
            print(
                'custom generation search:',
                f'candidates={len(evaluations)}',
                f'workers={workers}',
                f'winner={int(winner_index) + 1}',
                f'score={float(score):.3f}',
                flush=True,
            )
            return rewritten
        finally:
            for candidate_index, candidate_root, candidate_result in candidates:
                if (
                    winner_checkpoint_acknowledged
                    and candidate_index == winner_index
                ):
                    pass
                elif winner_live_committed and candidate_index != winner_index:
                    self.discard_generation_checkpoint(candidate_result)
                else:
                    self.release_generation_checkpoint(candidate_result)
                shutil.rmtree(candidate_root.parent, ignore_errors=True)
            shutil.rmtree(base_root, ignore_errors=True)

    generate_with_search.__signature__ = _public_signature_without_target_defaults(original)
    generate_with_search._mmm_parallel_custom_search = True
    generate_with_search._mmm_custom_verifier_search = True
    generate_with_search._mmm_research_generation_search = True
    cls.generate = generate_with_search


__all__ = [
    "_STRATEGIES",
    "_ResearchEvidenceRouter",
    "_StrategyRouter",
    "_active_native_slots",
    "_candidate_patch_capture",
    "_capture_candidate",
    "_fork_router_for_candidate",
    "_target_values",
    "_verify_candidate",
    "_width",
    "install",
]
