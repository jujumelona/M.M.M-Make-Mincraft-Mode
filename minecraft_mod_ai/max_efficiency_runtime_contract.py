from __future__ import annotations

"""Late runtime policy for compile, routing, and isolated candidate efficiency.

The production graph owns dependency safety, durable leases and executor capacity.
This contract keeps only optimizations that belong outside that scheduler owner:
compile-local reuse, exact integration routing, batched ledger status reads, and
isolated parallel custom candidate generation with a single winner commit.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Mapping


_FORCE_SINGLE_CUSTOM_SEARCH: ContextVar[bool] = ContextVar(
    "mmm_force_single_custom_search",
    default=False,
)
_WORK_GRAPH_HASH_CACHE: ContextVar[dict[int, str] | None] = ContextVar(
    "mmm_work_graph_hash_cache",
    default=None,
)
_WORK_GRAPH_VALIDATION_CACHE: ContextVar[set[tuple[int, int]] | None] = ContextVar(
    "mmm_work_graph_validation_cache",
    default=None,
)


def _active_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _install_work_graph_compile_cache(work_graph_module: Any) -> None:
    """Deduplicate validation and full proposal hashing inside one graph compile.

    CompleteProposal.validate already validates every proposal-owned module before
    build_production_work_plan historically validates that exact same object set a
    second time. The builder also asks for the same canonical proposal hash three
    times. Keep those checks authoritative on every invocation, but reuse their
    successful result only for the lifetime of that one synchronous compile. The
    ContextVars make concurrent compiles independent and prevent stale cross-call
    validation or hash reuse after callers mutate nested proposal data.
    """

    proposal_cls = work_graph_module.CompleteProposal
    module_cls = work_graph_module.ProductionModule

    current_hash = proposal_cls.calculate_hash
    if not getattr(current_hash, "_mmm_compile_local_hash_cache", False):

        @wraps(current_hash)
        def calculate_hash(self: Any) -> str:
            cache = _WORK_GRAPH_HASH_CACHE.get()
            if cache is None:
                return current_hash(self)
            key = id(self)
            if key not in cache:
                cache[key] = current_hash(self)
            return cache[key]

        calculate_hash._mmm_compile_local_hash_cache = True  # type: ignore[attr-defined]
        proposal_cls.calculate_hash = calculate_hash

    current_validate = module_cls.validate
    if not getattr(current_validate, "_mmm_compile_local_validation_cache", False):

        @wraps(current_validate)
        def validate(self: Any, *, policy: Any = None) -> None:
            cache = _WORK_GRAPH_VALIDATION_CACHE.get()
            if cache is None:
                current_validate(self, policy=policy)
                return
            key = (id(self), id(policy))
            if key in cache:
                return
            current_validate(self, policy=policy)
            cache.add(key)

        validate._mmm_compile_local_validation_cache = True  # type: ignore[attr-defined]
        module_cls.validate = validate

    current_build = work_graph_module.build_production_work_plan
    if getattr(current_build, "_mmm_compile_local_cache", False):
        return

    @wraps(current_build)
    def build_production_work_plan(*args: Any, **kwargs: Any):
        hash_token = _WORK_GRAPH_HASH_CACHE.set({})
        validation_token = _WORK_GRAPH_VALIDATION_CACHE.set(set())
        try:
            return current_build(*args, **kwargs)
        finally:
            _WORK_GRAPH_VALIDATION_CACHE.reset(validation_token)
            _WORK_GRAPH_HASH_CACHE.reset(hash_token)

    build_production_work_plan._mmm_compile_local_cache = True  # type: ignore[attr-defined]
    work_graph_module.build_production_work_plan = build_production_work_plan

    for loaded in tuple(sys.modules.values()):
        if loaded is None:
            continue
        try:
            if getattr(loaded, "build_production_work_plan", None) is current_build:
                setattr(loaded, "build_production_work_plan", build_production_work_plan)
        except (AttributeError, TypeError):
            continue


def _install_work_ledger_read_batching(work_graph_module: Any) -> None:
    """Remove N+1 SQLite reads from status pages without changing ledger authority."""

    ledger_cls = work_graph_module.DurableWorkLedger
    current_tasks = ledger_cls.tasks
    if not getattr(current_tasks, "_mmm_batched_status_reads", False):

        @wraps(current_tasks)
        def tasks(
            self: Any,
            *,
            cursor: str = "",
            limit: int = 100,
            state: Any = None,
        ) -> dict[str, Any]:
            if not 1 <= limit <= 1000:
                raise work_graph_module.WorkGraphError(
                    "Task page size must be between 1 and 1000."
                )
            clauses = ["node_id > ?"]
            params: list[Any] = [cursor]
            if state is not None:
                clauses.append("state = ?")
                params.append(state.value)
            params.append(limit + 1)
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT node_id, stage, input_hash, payload_json, state,
                           attempt, lease_owner, lease_until, output_hash,
                           receipt_json, error, updated_at
                    FROM tasks
                    WHERE {' AND '.join(clauses)}
                    ORDER BY node_id LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
                page_rows = rows[:limit]
                node_ids = [str(row[0]) for row in page_rows]
                dependencies: dict[str, list[str]] = {
                    node_id: [] for node_id in node_ids
                }
                for start in range(0, len(node_ids), 900):
                    batch = node_ids[start : start + 900]
                    if not batch:
                        continue
                    placeholders = ",".join("?" for _ in batch)
                    for node_id, dependency_id in connection.execute(
                        f"""
                        SELECT node_id, dependency_id FROM edges
                        WHERE node_id IN ({placeholders})
                        ORDER BY node_id, dependency_id
                        """,
                        tuple(batch),
                    ):
                        dependencies[str(node_id)].append(str(dependency_id))

            page = [
                {
                    "node_id": row[0],
                    "stage": row[1],
                    "input_hash": row[2],
                    "payload": json.loads(row[3]),
                    "state": row[4],
                    "attempt": row[5],
                    "lease_owner": row[6],
                    "lease_until": row[7],
                    "output_hash": row[8],
                    "receipt": json.loads(row[9]) if row[9] else None,
                    "error": row[10],
                    "updated_at": row[11],
                    "dependencies": dependencies[str(row[0])],
                }
                for row in page_rows
            ]
            return {
                "schema_version": "mmm/work-task-page-v1",
                "tasks": page,
                "next_cursor": page[-1]["node_id"] if len(rows) > limit else "",
            }

        tasks._mmm_batched_status_reads = True  # type: ignore[attr-defined]
        tasks.__wrapped__ = current_tasks  # type: ignore[attr-defined]
        ledger_cls.tasks = tasks

    current_summary = ledger_cls.summary
    if getattr(current_summary, "_mmm_single_connection_summary", False):
        return

    @wraps(current_summary)
    def summary(self: Any) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT state, COUNT(*) FROM tasks GROUP BY state"
                )
            }
            total = sum(counts.values())
            checkpoint_counts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT state, COUNT(*) FROM checkpoints GROUP BY state"
                )
            }
            cancel_requested = self._meta(connection, "cancel_requested")
            graph_hash = self._meta(connection, "graph_hash")
            module_count = self._meta(connection, "module_count")
        completed = counts.get(work_graph_module.WorkState.SUCCEEDED.value, 0)
        return {
            "schema_version": "mmm/work-ledger-summary-v1",
            "proposal_hash": self.proposal_hash,
            "graph_hash": graph_hash,
            "module_count": int(module_count or "0"),
            "task_count": total,
            "counts": counts,
            "checkpoint_counts": checkpoint_counts,
            "cancel_requested": cancel_requested or None,
            "progress": 1.0 if total == 0 else round(completed / total, 6),
        }

    summary._mmm_single_connection_summary = True  # type: ignore[attr-defined]
    summary.__wrapped__ = current_summary  # type: ignore[attr-defined]
    ledger_cls.summary = summary


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
                f"Custom candidate patch receipt has invalid operation for {relative}: {operation!r}"
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
                    f"Custom candidate after hash drifted for {relative}: {actual_after} != {after_sha}"
                )
            operations.append(
                {"operation": "create", "path": relative, "content": candidate_bytes.decode("utf-8")}
            )
            continue

        if base_bytes is None:
            raise RuntimeError(f"Custom candidate base file is missing: {relative}")
        actual_before = _sha256_receipt(base_bytes)
        if str(before_sha) != actual_before:
            raise RuntimeError(
                f"Custom candidate base hash drifted for {relative}: {actual_before} != {before_sha}"
            )

        if operation == "delete":
            if after_sha is not None:
                raise RuntimeError(f"Custom candidate delete has an after hash: {relative}")
            if candidate_path.exists() or candidate_bytes is not None:
                raise RuntimeError(f"Custom candidate delete output still exists: {relative}")
            operations.append(
                {"operation": "delete", "path": relative, "expected_sha256": actual_before}
            )
            continue

        if candidate_bytes is None:
            raise RuntimeError(f"Custom candidate replacement output is missing: {relative}")
        actual_after = _sha256_receipt(candidate_bytes)
        if str(after_sha) != actual_after:
            raise RuntimeError(
                f"Custom candidate after hash drifted for {relative}: {actual_after} != {after_sha}"
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
        tempfile.mkdtemp(prefix=f"candidate-{candidate_index:02d}-", dir=base_root.parent)
    ).resolve()
    candidate_root = workspace / "project"
    shutil.copytree(
        base_root,
        candidate_root,
        copy_function=performance_module._reflink_or_copy,
    )
    return candidate_root


def _install_parallel_custom_search(custom_module_generator_module: Any) -> None:
    """Parallelize candidate generation while the inner owner installs research once."""

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
            worker.router = search_module._fork_router_for_candidate(self.router)
            return current(worker, project_root, *args, **kwargs)

        live_root = Path(project_root).expanduser().resolve()
        base_root = performance_module._clone_source_snapshot(live_root)
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        errors: dict[int, BaseException] = {}

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
                search_module._fork_router_for_candidate(self.router),
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
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mmm_custom_generate",
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
                raise RuntimeError("Custom generation search produced no candidate.")

            def verify(item: tuple[int, Path, dict[str, Any]]):
                index, candidate_root, result = item
                score, verifier = search_module._verify_candidate(candidate_root, result)
                return score, index, candidate_root, result, verifier

            if len(candidates) == 1:
                evaluations = [verify(candidates[0])]
            else:
                with ThreadPoolExecutor(
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
                "schema_version": "mmm/custom-generation-search-v3",
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
                "research_aware": True,
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
    generate._mmm_research_generation_search = True  # type: ignore[attr-defined]
    cls.generate = generate


def enhance_runtime(*, work_graph_module: Any) -> None:
    """Install non-scheduler throughput features after runtime safety contracts."""

    from . import custom_module_generator

    _install_work_graph_compile_cache(work_graph_module)
    _install_work_ledger_read_batching(work_graph_module)
    _install_module_routing(work_graph_module)
    _install_parallel_custom_search(custom_module_generator)


__all__ = [
    "_active_parallelism",
    "_install_work_graph_compile_cache",
    "_install_work_ledger_read_batching",
    "_install_module_routing",
    "_install_parallel_custom_search",
    "enhance_runtime",
]
