from __future__ import annotations

"""Consolidate residual hot-path admission without duplicating work owners.

The existing bottleneck contract owns MCP transport/session reuse, central research
owns evidence semantics, and trajectory-memory performance owns its SQLite index.
This module only adjusts admission around those owners: duplicate central retrievals
are memoized, independent read-only MCP requests use bounded persistent lanes, and
trajectory memory locks are scoped to the project instead of serializing unrelated
projects in one process.
"""

import hashlib
import json
import os
import threading
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

_INSTALL_LOCK = threading.RLock()
_MCP_REQUEST_LANE: ContextVar[int | None] = ContextVar(
    "mmm_external_mcp_read_lane",
    default=None,
)
_MEMORY_BASE: ContextVar[str] = ContextVar("mmm_trajectory_memory_base", default="")
_MARKER = "_mmm_runtime_hotpath_consolidation_v1"
_MEMORY_MARKER = "_mmm_project_scoped_trajectory_memory_v1"
_DEFAULT_EXTERNAL_READ_LANES = 4
_MAX_EXTERNAL_READ_LANES = 8
_MAX_READ_CACHE_ENTRIES = 512


class _ProjectScopedRLock:
    """One reentrant lock per active project, with a safe global fallback."""

    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._locks: dict[str, threading.RLock] = {}
        self._local = threading.local()

    def _lock(self) -> threading.RLock:
        key = _MEMORY_BASE.get() or "__global__"
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def __enter__(self) -> "_ProjectScopedRLock":
        lock = self._lock()
        lock.acquire()
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        stack.append(lock)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        stack = getattr(self._local, "stack", None)
        if not stack:
            raise RuntimeError("Project-scoped trajectory lock exit without enter")
        stack.pop().release()


def _external_read_lanes() -> int:
    raw = os.environ.get("MMM_EXTERNAL_MCP_READ_LANES", "").strip()
    if not raw:
        return _DEFAULT_EXTERNAL_READ_LANES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_EXTERNAL_READ_LANES
    return max(1, min(_MAX_EXTERNAL_READ_LANES, value))


def _request_lane(
    server_name: str,
    tool: str,
    arguments: Mapping[str, Any],
) -> int:
    lanes = _external_read_lanes()
    if lanes <= 1:
        return 0
    rendered = json.dumps(
        {
            "server": server_name,
            "tool": tool,
            "arguments": dict(arguments),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(rendered.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % lanes


def _trim_external_read_cache(bottlenecks: Any) -> None:
    cache = getattr(bottlenecks, "_READ_CACHE", None)
    lock = getattr(bottlenecks, "_READ_LOCK", None)
    if not isinstance(cache, dict) or lock is None:
        return
    with lock:
        overflow = len(cache) - _MAX_READ_CACHE_ENTRIES
        for _ in range(max(0, overflow)):
            try:
                oldest = next(iter(cache))
            except StopIteration:
                return
            cache.pop(oldest, None)


def _install_external_mcp_admission(bottlenecks: Any, external: Any) -> None:
    current_worker = bottlenecks._external_worker
    if not getattr(current_worker, "_mmm_deterministic_read_lanes", False):

        @wraps(current_worker)
        def lane_worker(router: Any, server_name: str, entry: Mapping[str, Any]):
            lane = _MCP_REQUEST_LANE.get()
            if lane is None:
                return current_worker(router, server_name, entry)
            worker_key_name = f"{server_name}#read-lane-{lane}"
            return current_worker(router, worker_key_name, entry)

        lane_worker._mmm_deterministic_read_lanes = True  # type: ignore[attr-defined]
        lane_worker.__wrapped__ = current_worker  # type: ignore[attr-defined]
        bottlenecks._external_worker = lane_worker

    current_call = external.ExternalMCPRouter._call_provider
    if getattr(current_call, "_mmm_deterministic_read_lanes", False):
        return

    @wraps(current_call)
    def admitted_call(
        self: Any,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        stage = bottlenecks._MCP_STAGE.get()
        read_only = stage != "runtime" and bottlenecks._read_only_tool(entry, tool)
        token = None
        if read_only:
            token = _MCP_REQUEST_LANE.set(
                _request_lane(server_name, tool, arguments)
            )
        try:
            return current_call(
                self,
                server_name,
                entry,
                tool=tool,
                arguments=arguments,
            )
        finally:
            if token is not None:
                _MCP_REQUEST_LANE.reset(token)
            _trim_external_read_cache(bottlenecks)

    admitted_call._mmm_deterministic_read_lanes = True  # type: ignore[attr-defined]
    admitted_call.__wrapped__ = current_call  # type: ignore[attr-defined]
    external.ExternalMCPRouter._call_provider = admitted_call


def _install_central_research_dedup(central_research: Any) -> Any:
    current = central_research.retrieve_domain_evidence
    if getattr(current, "_mmm_per_build_retrieval_dedup", False):
        return current

    defaults = getattr(current, "__kwdefaults__", None) or {}
    default_retrieve = defaults.get(
        "retrieve",
        central_research.retrieve_official_evidence,
    )

    @wraps(current)
    def deduplicated_retrieve_domain_evidence(
        research_brief: dict[str, Any],
        *,
        retrieve: Any = default_retrieve,
    ) -> dict[str, Any]:
        receipts: dict[str, Any] = {}
        receipts_lock = threading.RLock()

        def retrieve_once(query: str, **kwargs: Any) -> Any:
            rendered = json.dumps(
                {
                    "query": str(query),
                    "kwargs": kwargs,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            key = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            with receipts_lock:
                cached = receipts.get(key)
            if cached is not None:
                return cached
            value = retrieve(query, **kwargs)
            with receipts_lock:
                return receipts.setdefault(key, value)

        return current(research_brief, retrieve=retrieve_once)

    deduplicated_retrieve_domain_evidence._mmm_per_build_retrieval_dedup = True  # type: ignore[attr-defined]
    deduplicated_retrieve_domain_evidence.__wrapped__ = current  # type: ignore[attr-defined]
    central_research.retrieve_domain_evidence = deduplicated_retrieve_domain_evidence
    return deduplicated_retrieve_domain_evidence


def harden_memory_lock_scope(
    trajectory_memory: Any,
    memory_performance: Any,
    indexed_append: Any,
    indexed_relevant: Any,
) -> tuple[Any, Any]:
    """Remove process-global memory waits while preserving same-project exclusion."""

    if getattr(indexed_append, _MEMORY_MARKER, False):
        return indexed_append, indexed_relevant

    keyed_lock = _ProjectScopedRLock()
    trajectory_memory._LOCK = keyed_lock
    memory_performance._TRAJECTORY_LOCK = keyed_lock

    @wraps(indexed_append)
    def append(base: str | Path, row: Mapping[str, Any]) -> bool:
        token = _MEMORY_BASE.set(str(Path(base).expanduser().resolve()))
        try:
            return bool(indexed_append(base, row))
        finally:
            _MEMORY_BASE.reset(token)

    @wraps(indexed_relevant)
    def relevant(
        base: str | Path,
        query: str,
        *,
        task_class: str,
        router: Any | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        token = _MEMORY_BASE.set(str(Path(base).expanduser().resolve()))
        try:
            return indexed_relevant(
                base,
                query,
                task_class=task_class,
                router=router,
                limit=limit,
            )
        finally:
            _MEMORY_BASE.reset(token)

    setattr(append, _MEMORY_MARKER, True)
    append.__wrapped__ = indexed_append  # type: ignore[attr-defined]
    setattr(relevant, _MEMORY_MARKER, True)
    relevant.__wrapped__ = indexed_relevant  # type: ignore[attr-defined]
    trajectory_memory.append_trajectory = append
    trajectory_memory.relevant_trajectories = relevant
    return append, relevant


def harden(
    bottlenecks: Any,
    central_research: Any,
    external_mcp_router: Any,
    agentic_research: Any,
) -> None:
    """Harden existing owners once; package bootstrap remains the only composer."""
    with _INSTALL_LOCK:
        if getattr(harden, _MARKER, False):
            return
        _install_external_mcp_admission(bottlenecks, external_mcp_router)
        shared_research = _install_central_research_dedup(central_research)
        agentic_research.retrieve_domain_evidence = shared_research
        setattr(harden, _MARKER, True)


__all__ = ["harden", "harden_memory_lock_scope"]
