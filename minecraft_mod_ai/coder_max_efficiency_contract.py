from __future__ import annotations

import copy
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Any

from .runtime_contract_wrappers import has_contract_marker, owns_contract_marker

_MARKER = "_mmm_coder_max_efficiency_v1"
_RESEARCH_SINGLEFLIGHT_MARKER = "_mmm_research_initial_bundle_singleflight_v1"
_RESEARCH_STRIPES = tuple(threading.Lock() for _ in range(32))


def _candidate_worker_count(count: int, active_slots: int) -> int:
    if count <= 1:
        return 1
    return max(1, min(int(count), max(1, int(active_slots))))


def _research_lock(cache_key: str) -> threading.Lock:
    return _RESEARCH_STRIPES[hash(cache_key) % len(_RESEARCH_STRIPES)]


def _install_research_singleflight(research_module: Any) -> None:
    cls = research_module.ResearchCodeContext
    current = cls.initial_bundle
    if getattr(current, _RESEARCH_SINGLEFLIGHT_MARKER, False):
        return

    @wraps(current)
    def initial_bundle(self: Any) -> dict[str, Any]:
        if bool(getattr(self, "_initial_complete", False)):
            return current(self)
        cache_key = str(self._initial_cache_key())
        with _research_lock(cache_key):
            return current(self)

    setattr(initial_bundle, _RESEARCH_SINGLEFLIGHT_MARKER, True)
    cls.initial_bundle = initial_bundle


def _clone_candidate_owner(owner: Any, search_module: Any) -> Any:
    candidate = copy.copy(owner)
    candidate.router = search_module._fork_router_for_candidate(owner.router)
    candidate._cached_index = None
    candidate._cached_root = None
    return candidate


def _clone_candidate_roots(
    performance_module: Any,
    root: Path,
    *,
    count: int,
) -> list[Path]:
    base_snapshot = performance_module._acquire_wave_source_snapshot(root)
    roots: list[Path] = []
    try:
        for _ in range(count):
            roots.append(performance_module._clone_wave_workspace(base_snapshot, root))
        return roots
    except BaseException:
        for candidate_root in roots:
            shutil.rmtree(candidate_root, ignore_errors=True)
        raise
    finally:
        performance_module._release_wave_source_snapshot(root, base_snapshot)


def _run_candidate_generation(
    owner: Any,
    single_generate: Callable[..., Any],
    candidate_roots: Sequence[Path],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    search_module: Any,
) -> tuple[list[tuple[int, Path, dict[str, Any], dict[str, Any]]], list[BaseException]]:
    count = len(candidate_roots)
    workers = _candidate_worker_count(count, search_module._active_native_slots())
    candidates: list[tuple[int, Path, dict[str, Any], dict[str, Any]]] = []
    errors: list[BaseException] = []

    def run(candidate_index: int, candidate_root: Path):
        candidate_owner = _clone_candidate_owner(owner, search_module)
        strategy = search_module._STRATEGIES[candidate_index % len(search_module._STRATEGIES)]
        return search_module._capture_candidate(
            candidate_owner,
            single_generate,
            candidate_root,
            strategy=strategy,
            candidate_index=candidate_index,
            count=count,
            args=args,
            kwargs=dict(kwargs),
        )

    if workers <= 1:
        for candidate_index, candidate_root in enumerate(candidate_roots):
            try:
                result, capture = run(candidate_index, candidate_root)
            except BaseException as exc:
                errors.append(exc)
                shutil.rmtree(candidate_root, ignore_errors=True)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            else:
                candidates.append((candidate_index, candidate_root, result, capture))
        return candidates, errors

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="mmm_custom_generate",
    ) as pool:
        pending: dict[Future[Any], tuple[int, Path]] = {
            pool.submit(run, candidate_index, candidate_root): (candidate_index, candidate_root)
            for candidate_index, candidate_root in enumerate(candidate_roots)
        }
        for future in as_completed(pending):
            candidate_index, candidate_root = pending[future]
            try:
                result, capture = future.result()
            except BaseException as exc:
                errors.append(exc)
                shutil.rmtree(candidate_root, ignore_errors=True)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            else:
                candidates.append((candidate_index, candidate_root, result, capture))

    candidates.sort(key=lambda item: item[0])
    return candidates, errors


def _evaluate_candidates(
    candidates: Sequence[tuple[int, Path, dict[str, Any], dict[str, Any]]],
    *,
    search_module: Any,
) -> list[tuple[float, int, Path, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    if len(candidates) == 1:
        candidate_index, candidate_root, result, capture = candidates[0]
        score, verifier = search_module._verify_candidate(candidate_root, result)
        return [(score, candidate_index, candidate_root, result, capture, verifier)]

    evaluations = []
    workers = min(2, len(candidates))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm_custom_verify") as pool:
        pending = [
            (
                candidate_index,
                candidate_root,
                result,
                capture,
                pool.submit(search_module._verify_candidate, candidate_root, result),
            )
            for candidate_index, candidate_root, result, capture in candidates
        ]
        for candidate_index, candidate_root, result, capture, future in pending:
            score, verifier = future.result()
            evaluations.append(
                (score, candidate_index, candidate_root, result, capture, verifier)
            )
    return evaluations


def _commit_winner(
    root: Path,
    winner_root: Path,
    capture: Mapping[str, Any],
    *,
    performance_module: Any,
    source_patch_module: Any,
) -> dict[str, Any]:
    operations = capture.get("operations", [])
    if not isinstance(operations, list) or not operations:
        raise RuntimeError("Winning custom candidate contains no patch operations.")

    from .project_write_lock import project_write_lock

    with project_write_lock(root):
        return performance_module._commit_staged_operations(
            live_root=root,
            staging_root=winner_root,
            capture=dict(capture),
            source_patch_module=source_patch_module,
        )


def _parallel_generate(
    owner: Any,
    single_generate: Callable[..., Any],
    project_root: str | Path,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    search_module: Any,
) -> dict[str, Any]:
    from . import performance_final_contract as performance_module
    from . import source_patch as source_patch_module

    root = Path(project_root).expanduser().resolve()
    count = search_module._width(kwargs.get("module"))
    candidate_roots = _clone_candidate_roots(performance_module, root, count=count)
    candidates: list[tuple[int, Path, dict[str, Any], dict[str, Any]]] = []
    winner_index: int | None = None
    winner_checkpoint_acknowledged = False
    winner_live_committed = False
    try:
        candidates, errors = _run_candidate_generation(
            owner,
            single_generate,
            candidate_roots,
            args=args,
            kwargs=kwargs,
            search_module=search_module,
        )
        if not candidates:
            if errors:
                raise errors[-1]
            raise RuntimeError("Custom generation search produced no candidate.")

        evaluations = _evaluate_candidates(candidates, search_module=search_module)
        evaluations.sort(
            key=lambda item: (
                -item[0],
                search_module._json_size(item[4].get("operations", [])),
                item[1],
            )
        )
        score, winner_index, winner_root, result, capture, verifier = evaluations[0]
        commit_receipt = _commit_winner(
            root,
            winner_root,
            capture,
            performance_module=performance_module,
            source_patch_module=source_patch_module,
        )
        winner_live_committed = True
        rewritten = performance_module._rewrite_root_paths(result, winner_root, root)
        rewritten["patch_receipt"] = commit_receipt
        rewritten["agentic_generation_search"] = {
            "schema_version": "mmm/custom-generation-search-v4",
            "candidate_count": len(evaluations),
            "winner_index": winner_index,
            "winner_score": score,
            "winner_verifier": verifier,
            "candidate_scores": [
                {
                    "candidate_index": item[1],
                    "score": item[0],
                    "verifier": item[5],
                }
                for item in sorted(evaluations, key=lambda item: item[1])
            ],
            "research_aware": True,
            "dependency_admission": "exact",
            "candidate_generation": "bounded_parallel",
            "candidate_workers": _candidate_worker_count(
                count,
                search_module._active_native_slots(),
            ),
            "snapshot_policy": "single_frozen_base_then_cow_forks",
            "research_initialization": "content_addressed_singleflight",
            "winner_commit": "three_way_rebase_preserving_concurrent_changes",
        }
        winner_checkpoint_acknowledged = bool(
            owner.acknowledge_generation_checkpoint(rewritten)
        )
        rewritten["generation_checkpoint_acknowledged"] = (
            winner_checkpoint_acknowledged
        )
        print(
            "custom generation search:",
            f"candidates={len(evaluations)}",
            f"winner={winner_index + 1}",
            f"score={score:.3f}",
            f"workers={rewritten['agentic_generation_search']['candidate_workers']}",
            flush=True,
        )
        return rewritten
    finally:
        for candidate_index, _candidate_root, candidate_result, _capture in candidates:
            if (
                winner_checkpoint_acknowledged
                and candidate_index == winner_index
            ):
                continue
            if winner_live_committed and candidate_index != winner_index:
                owner.discard_generation_checkpoint(candidate_result)
            else:
                owner.release_generation_checkpoint(candidate_result)
        for candidate_root in candidate_roots:
            shutil.rmtree(candidate_root, ignore_errors=True)


def install_coder_max_efficiency() -> None:
    """Replace only the serial multi-candidate coder path with bounded parallel search."""

    from . import custom_generation_search_contract as search_module
    from . import research_code_context as research_module
    from .custom_module_generator import CustomModuleGenerator

    _install_research_singleflight(research_module)

    current = CustomModuleGenerator.generate
    if has_contract_marker(current, _MARKER):
        return
    single_generate = getattr(current, "__wrapped__", None)
    if not (
        owns_contract_marker(current, "_mmm_research_generation_search")
        and callable(single_generate)
    ):
        raise RuntimeError(
            "Coder max-efficiency contract requires the reviewed custom generation search wrapper."
        )

    @wraps(current)
    def generate(
        self: Any,
        project_root: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        count = search_module._width(kwargs.get("module"))
        if count <= 1:
            result = current(self, project_root, *args, **kwargs)
            if isinstance(result, dict):
                result["generation_checkpoint_acknowledged"] = bool(
                    self.acknowledge_generation_checkpoint(result)
                )
            return result
        return _parallel_generate(
            self,
            single_generate,
            project_root,
            args=args,
            kwargs=dict(kwargs),
            search_module=search_module,
        )

    for name, value in vars(current).items():
        if name.startswith("_mmm_") and value is True:
            setattr(generate, name, True)
    setattr(generate, _MARKER, True)
    generate._mmm_parallel_candidate_generation = True
    generate._mmm_single_snapshot_candidate_forks = True
    generate._mmm_research_initial_bundle_singleflight = True
    generate._mmm_concurrent_winner_rebase = True
    if hasattr(current, "__signature__"):
        generate.__signature__ = current.__signature__
    CustomModuleGenerator.generate = generate


__all__ = [
    "_candidate_worker_count",
    "_install_research_singleflight",
    "_run_candidate_generation",
    "install_coder_max_efficiency",
]
