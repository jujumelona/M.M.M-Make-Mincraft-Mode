from __future__ import annotations

"""Late runtime policy that makes declared parallel capacity real end to end.

The production graph already owns dependency safety, durable leases and narrow commits.
This contract removes the remaining execution bottlenecks without weakening those
boundaries: exact executor capacity, exact integration routing, and isolated parallel
custom candidate generation with a single winner commit. Scheduler claim ownership
stays exclusively in scheduler_parallel_safety_contract.
"""

import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
import tempfile
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Mapping


_FORCE_SINGLE_CUSTOM_SEARCH: ContextVar[bool] = ContextVar(
    "mmm_force_single_custom_search",
    default=False,
)
_ORIGINAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor


def _active_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _install_exact_llm_executor() -> None:
    """Make the production ``llm`` pool match the native server slot capacity."""

    current = concurrent.futures.ThreadPoolExecutor
    if getattr(current, "_mmm_exact_llm_executor", False):
        return
    base = current

    class ExactCapacityThreadPoolExecutor(base):
        def __init__(
            self,
            max_workers: int | None = None,
            thread_name_prefix: str = "",
            initializer: Any = None,
            initargs: tuple[Any, ...] = (),
        ) -> None:
            if thread_name_prefix == "llm":
                max_workers = _active_parallelism()
            super().__init__(
                max_workers=max_workers,
                thread_name_prefix=thread_name_prefix,
                initializer=initializer,
                initargs=initargs,
            )

    ExactCapacityThreadPoolExecutor._mmm_exact_llm_executor = True  # type: ignore[attr-defined]
    ExactCapacityThreadPoolExecutor._mmm_base_executor = base  # type: ignore[attr-defined]
    concurrent.futures.ThreadPoolExecutor = ExactCapacityThreadPoolExecutor


def _install_module_routing(work_graph_module: Any) -> None:
    """Route only genuinely model-backed integrations into the custom LLM lane."""

    current_stage = work_graph_module._module_stage
    if getattr(current_stage, "_mmm_exact_integration_stage", False):
        return

    @wraps(current_stage)
    def module_stage(module: Any) -> str:
        from .research_ledger import is_research_shard

        if is_research_shard(module) or str(getattr(module, "kind", "")) == "research_shard":
            return current_stage(module)
        if str(getattr(module, "kind", "")) == "integration":
            config = getattr(module, "config", {})
            config = config if isinstance(config, Mapping) else {}
            if str(config.get("integration_type", "")) == "mmm_local_ai_sidecar":
                return "content"
            return "custom"
        return current_stage(module)

    module_stage._mmm_exact_integration_stage = True  # type: ignore[attr-defined]
    work_graph_module._module_stage = module_stage


def _sha256_receipt(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _candidate_patch_capture(
    *,
    base_root: Path,
    candidate_root: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild one winning patch only after exact receipt/file verification."""

    receipt = result.get("patch_receipt")
    receipt_ops = receipt.get("operations") if isinstance(receipt, Mapping) else None
    if not isinstance(receipt_ops, list) or not receipt_ops:
        raise RuntimeError("Custom candidate has no staged patch receipt.")

    operations: list[dict[str, Any]] = []
    before: dict[str, bytes | None] = {}
    seen_paths: set[str] = set()
    for item in receipt_ops:
        if not isinstance(item, Mapping):
            raise RuntimeError("Custom candidate patch receipt is malformed.")
        relative = str(item.get("path", "")).strip()
        operation = str(item.get("operation", "")).strip().lower()
        if not relative:
            raise RuntimeError("Custom candidate patch receipt has an empty path.")
        if relative in seen_paths:
            raise RuntimeError(f"Custom candidate patch receipt repeats path: {relative}")
        seen_paths.add(relative)
        if operation not in {"create", "replace", "edit", "delete"}:
            raise RuntimeError(
                f"Custom candidate patch receipt has invalid operation for {relative}: "
                f"{operation!r}"
            )

        base_path = (base_root / relative).resolve()
        candidate_path = (candidate_root / relative).resolve()
        try:
            base_path.relative_to(base_root)
            candidate_path.relative_to(candidate_root)
        except ValueError as exc:
            raise RuntimeError(f"Custom candidate path escaped staging root: {relative}") from exc
        if base_path.is_symlink() or candidate_path.is_symlink():
            raise RuntimeError(f"Custom candidate patch path may not be a symlink: {relative}")

        base_bytes = base_path.read_bytes() if base_path.is_file() else None
        candidate_bytes = candidate_path.read_bytes() if candidate_path.is_file() else None
        before[relative] = base_bytes
        before_sha = item.get("before_sha256")
        after_sha = item.get("after_sha256")

        if operation == "create":
            if before_sha is not None or base_bytes is not None:
                raise RuntimeError(f"Custom candidate create precondition is invalid: {relative}")
            if candidate_bytes is None:
                raise RuntimeError(f"Custom candidate create output is missing: {relative}")
            actual_after = _sha256_receipt(candidate_bytes)
            if str(after_sha) != actual_after:
                raise RuntimeError(
                    f"Custom candidate after hash drifted for {relative}: "
                    f"{actual_after} != {after_sha}"
                )
            operations.append(
                {
                    "operation": "create",
                    "path": relative,
                    "content": candidate_bytes.decode("utf-8"),
                }
            )
            continue

        if base_bytes is None:
            raise RuntimeError(f"Custom candidate base file is missing: {relative}")
        actual_before = _sha256_receipt(base_bytes)
        if str(before_sha) != actual_before:
            raise RuntimeError(
                f"Custom candidate base hash drifted for {relative}: "
                f"{actual_before} != {before_sha}"
            )

        if operation == "delete":
            if after_sha is not None:
                raise RuntimeError(f"Custom candidate delete has an after hash: {relative}")
            if candidate_path.exists() or candidate_bytes is not None:
                raise RuntimeError(f"Custom candidate delete output still exists: {relative}")
            operations.append(
                {
                    "operation": "delete",
                    "path": relative,
                    "expected_sha256": actual_before,
                }
            )
            continue

        if candidate_bytes is None:
            raise RuntimeError(f"Custom candidate replacement output is missing: {relative}")
        actual_after = _sha256_receipt(candidate_bytes)
        if str(after_sha) != actual_after:
            raise RuntimeError(
                f"Custom candidate after hash drifted for {relative}: "
                f"{actual_after} != {after_sha}"
            )
        operations.append(
            {
                "operation": "replace",
                "path": relative,
                "expected_sha256": actual_before,
                "content": candidate_bytes.decode("utf-8"),
            }
        )
    return {"operations": operations, "before": before}


def _clone_candidate_snapshot(
    base_root: Path,
    *,
    candidate_index: int,
    performance_module: Any,
) -> Path:
    """Clone one candidate under its own workspace/RAG parent directory."""

    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"candidate-{candidate_index:02d}-",
            dir=base_root.parent,
        )
    ).resolve()
    candidate_root = workspace / "project"
    shutil.copytree(
        base_root,
        candidate_root,
        copy_function=performance_module._reflink_or_copy,
    )
    return candidate_root


def _install_parallel_custom_search(custom_module_generator_module: Any) -> None:
    """Parallelize the expensive custom candidate generation, not just verification."""

    from . import custom_generation_search_contract as search_module
    from . import performance_final_contract as performance_module
    from . import source_patch as source_patch_module

    current_width = search_module._width
    if not getattr(current_width, "_mmm_context_single_candidate", False):

        @wraps(current_width)
        def width(module: Any) -> int:
            if _FORCE_SINGLE_CUSTOM_SEARCH.get():
                return 1
            return current_width(module)

        width._mmm_context_single_candidate = True  # type: ignore[attr-defined]
        width.__wrapped__ = current_width  # type: ignore[attr-defined]
        search_module._width = width

    cls = custom_module_generator_module.CustomModuleGenerator
    current = cls.generate
    if getattr(current, "_mmm_max_parallel_custom_search", False):
        return

    @wraps(current)
    def generate(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        module = kwargs.get("module")
        count = int(search_module._width(module))

        if count <= 1:
            worker = copy.copy(self)
            worker._cached_index = None
            worker._cached_root = None
            worker.router = search_module._host_evidence_router(
                search_module._fork_router_for_candidate(self.router)
            )
            return current(worker, project_root, *args, **kwargs)

        live_root = Path(project_root).expanduser().resolve()
        base_root = performance_module._clone_source_snapshot(live_root)
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        errors: dict[int, Exception] = {}

        def solve(candidate_index: int) -> tuple[int, Path, dict[str, Any]]:
            candidate_root = _clone_candidate_snapshot(
                base_root,
                candidate_index=candidate_index,
                performance_module=performance_module,
            )
            worker = copy.copy(self)
            worker._cached_index = None
            worker._cached_root = None
            strategy = search_module._STRATEGIES[
                candidate_index % len(search_module._STRATEGIES)
            ]
            worker.router = search_module._StrategyRouter(
                search_module._host_evidence_router(
                    search_module._fork_router_for_candidate(self.router)
                ),
                strategy=strategy,
                candidate_index=candidate_index,
                count=count,
            )
            token = _FORCE_SINGLE_CUSTOM_SEARCH.set(True)
            try:
                result = current(worker, candidate_root, *args, **kwargs)
                if not isinstance(result, dict):
                    raise RuntimeError("Custom generation candidate returned a non-object receipt.")
                return candidate_index, candidate_root, result
            except BaseException:
                shutil.rmtree(candidate_root.parent, ignore_errors=True)
                raise
            finally:
                _FORCE_SINGLE_CUSTOM_SEARCH.reset(token)

        try:
            workers = min(count, _active_parallelism())
            with _ORIGINAL_THREAD_POOL(
                max_workers=workers,
                thread_name_prefix="mmm_custom_generate",
            ) as pool:
                futures = [pool.submit(solve, index) for index in range(count)]
                for candidate_index, future in enumerate(futures):
                    try:
                        candidates.append(future.result())
                    except Exception as exc:
                        errors[candidate_index] = exc
            candidates.sort(key=lambda item: item[0])
            if not candidates:
                if errors:
                    raise errors[max(errors)]
                raise RuntimeError("Custom generation search produced no candidate.")

            def verify(item: tuple[int, Path, dict[str, Any]]):
                index, candidate_root, result = item
                score, verifier = search_module._verify_candidate(candidate_root, result)
                return score, index, candidate_root, result, verifier

            if len(candidates) == 1:
                evaluations = [verify(candidates[0])]
            else:
                with _ORIGINAL_THREAD_POOL(
                    max_workers=min(2, len(candidates)),
                    thread_name_prefix="mmm_custom_verify",
                ) as pool:
                    evaluations = list(pool.map(verify, candidates))

            evaluations.sort(
                key=lambda item: (
                    -float(item[0]),
                    len(json.dumps(item[3], ensure_ascii=False, sort_keys=True)),
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
            rewritten = performance_module._rewrite_root_paths(result, winner_root, live_root)
            rewritten["patch_receipt"] = commit_receipt
            rewritten["agentic_generation_search"] = {
                "schema_version": "mmm/custom-generation-search-v2",
                "candidate_count": len(evaluations),
                "candidate_workers": workers,
                "winner_index": int(winner_index),
                "winner_score": float(score),
                "winner_verifier": verifier,
                "candidate_scores": [
                    {
                        "candidate_index": int(item[1]),
                        "score": float(item[0]),
                        "verifier": item[4],
                    }
                    for item in sorted(evaluations, key=lambda value: value[1])
                ],
            }
            print(
                "custom generation search:",
                f"candidates={len(evaluations)}",
                f"workers={workers}",
                f"winner={int(winner_index) + 1}",
                f"score={float(score):.3f}",
                flush=True,
            )
            return rewritten
        finally:
            for _index, candidate_root, _result in candidates:
                shutil.rmtree(candidate_root.parent, ignore_errors=True)
            shutil.rmtree(base_root, ignore_errors=True)

    generate._mmm_max_parallel_custom_search = True  # type: ignore[attr-defined]
    generate._mmm_custom_verifier_search = True  # type: ignore[attr-defined]
    generate._mmm_host_evidence_router = True  # type: ignore[attr-defined]
    cls.generate = generate


def enhance_runtime(*, work_graph_module: Any, scheduler_module: Any) -> None:
    """Install throughput features after scheduler and llama safety contracts."""

    from . import custom_module_generator

    del scheduler_module
    _install_exact_llm_executor()
    _install_module_routing(work_graph_module)
    _install_parallel_custom_search(custom_module_generator)


__all__ = [
    "_active_parallelism",
    "_install_exact_llm_executor",
    "_install_module_routing",
    "_install_parallel_custom_search",
    "enhance_runtime",
]
