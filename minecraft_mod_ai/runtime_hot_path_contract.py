from __future__ import annotations

"""Late hot-path fixes that preserve existing runtime contracts.

The composed runtime already owns schema integrity, provider binding and transport
schema identity. This module removes three performance hazards that remain beneath
those contracts without weakening their safety boundaries:

* MCP worker enqueue/startup must never block the caller's asyncio event loop.
* Independent external MCP providers must not share one global execution lock.
* Semantic RAG queries must not rebuild/reconcile the LSH side index on every query.

The LSH path stays fail-safe: every semantic build invalidates the ready marker before
mutation, the canonical reconciliation path publishes readiness only after it
completes, and searches without a ready marker raise so callers fall back to the
canonical exhaustive retrieval path.
"""

import asyncio
import concurrent.futures
import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

from .root_cause_trace import emit_root_cause

_MCP_MARKER = "_mmm_nonblocking_transport_execute_v1"
_EXTERNAL_MARKER = "_mmm_external_mcp_parallel_provider_v1"
_RAG_MARKER = "_mmm_rag_lsh_query_ready_v1"
_LSH_STATE_TABLE = "mmm_semantic_lsh_state"
_LSH_STATE_KEY = "ready"
_LSH_STATE_VERSION = "v1"
_SQLITE_MAGIC = b"SQLite format 3\x00"
_MCP_MAX_CONSECUTIVE_INFRA_FAILURES = 3
_MCP_RETRY_SAFE_VERIFIERS = frozenset(
    {
        "java_diagnostics",
        "jdt_diagnostics",
        "run_gradle_build",
        "gradle_build",
        "run_gametest",
    }
)
_MCP_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection closed",
    "channel closed",
    "transport closed",
    "session closed",
    "closed resource",
    "broken pipe",
    "broken resource",
    "end of stream",
    "unexpected eof",
    "eof",
)
_MCP_DETERMINISTIC_MARKERS = (
    "invalid task diagnostic path",
    "relative_files must be",
    "argument",
    "schema drift",
    "schema",
    "unsupported",
    "unsafe path",
    "outside",
    "no such file",
    "not found",
    "does not exist",
    "no java files",
)


def _exception_objects(exc: BaseException) -> tuple[BaseException, ...]:
    seen: set[int] = set()
    pending = [exc]
    result: list[BaseException] = []
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)
    return tuple(result)


def _is_transient_mcp_transport_failure(exc: BaseException) -> bool:
    chain = _exception_objects(exc)
    if any(isinstance(item, asyncio.CancelledError) for item in chain):
        return False
    if any(
        isinstance(item, (TimeoutError, ConnectionError, EOFError, BrokenPipeError))
        for item in chain
    ):
        return True
    text = " | ".join(
        f"{type(item).__name__}: {item}" for item in chain
    ).casefold()
    if any(marker in text for marker in _MCP_DETERMINISTIC_MARKERS):
        return False
    return any(marker in text for marker in _MCP_TRANSIENT_MARKERS)


async def _submit_without_blocking_loop(worker: Any, request: Any) -> None:
    """Apply bounded queue backpressure without blocking the caller event loop."""

    request.result.add_done_callback(lambda _future: worker._release_pending())
    enqueue: concurrent.futures.Future[Any] | None = None
    try:
        await asyncio.to_thread(worker._ensure_started)
        with worker._state_lock:
            if worker._closed:
                raise RuntimeError("MCP transport worker is closed")
            startup_error = worker._startup_error
            loop = worker._loop
            queue = worker._queue
        if startup_error is not None:
            raise RuntimeError("MCP transport worker failed to start") from startup_error
        if loop is None or queue is None:
            raise RuntimeError("MCP transport worker did not initialize")

        enqueue = asyncio.run_coroutine_threadsafe(queue.put(request), loop)
        await asyncio.wait_for(
            asyncio.wrap_future(enqueue),
            timeout=request.timeout_seconds,
        )
    except BaseException:
        if enqueue is not None and not enqueue.done():
            enqueue.cancel()
        if not request.result.done():
            request.result.cancel()
        raise


def _install_nonblocking_transport(mcp_transport_pool_module: Any) -> None:
    pool_class = mcp_transport_pool_module.MCPTransportPool
    current = pool_class._execute
    if bool(getattr(current, _MCP_MARKER, False)):
        return

    async def execute(
        self: Any,
        *,
        operation: str,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
        name: str = "",
        arguments: Mapping[str, Any] | None = None,
        expected_schema_sha256: str = "",
    ) -> Any:
        retry_safe = operation == "call_tool" and name in _MCP_RETRY_SAFE_VERIFIERS
        failures: list[dict[str, Any]] = []
        while True:
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            request = mcp_transport_pool_module._TransportRequest(
                operation=operation,
                stage=stage,
                env=dict(env),
                timeout_seconds=float(timeout_seconds),
                result=future,
                name=name,
                arguments=dict(arguments or {}),
                expected_schema_sha256=expected_schema_sha256,
            )
            worker = self._reserve_worker()
            try:
                await _submit_without_blocking_loop(worker, request)
                value = await asyncio.wrap_future(future)
            except BaseException as exc:
                if not retry_safe or not _is_transient_mcp_transport_failure(exc):
                    raise
                failures.append(
                    {
                        "attempt": len(failures) + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                exhausted = len(failures) >= _MCP_MAX_CONSECUTIVE_INFRA_FAILURES
                emit_root_cause(
                    "mcp_verifier_transport_retry",
                    stage=stage,
                    operation=name,
                    gate="mcp_transport_health",
                    result="FAIL" if exhausted else "RETRY",
                    reason=(
                        "transient verifier transport retries exhausted"
                        if exhausted
                        else "transient verifier transport failure; worker session will be reacquired"
                    ),
                    details={
                        "failure_count": len(failures),
                        "max_failures": _MCP_MAX_CONSECUTIVE_INFRA_FAILURES,
                        "retry_history": tuple(failures),
                    },
                    exc=exc,
                )
                if exhausted:
                    from .agent_tool_runtime import AgentToolRuntimeError

                    raise AgentToolRuntimeError(
                        "MCP_TRANSIENT_RETRY_EXHAUSTED: verifier transport failed "
                        f"{_MCP_MAX_CONSECUTIVE_INFRA_FAILURES} consecutive times; "
                        f"retry_history={json.dumps(failures, ensure_ascii=False, sort_keys=True)}"
                    ) from exc
                await asyncio.sleep(0)
                continue

            if failures:
                emit_root_cause(
                    "mcp_verifier_transport_recovered",
                    stage=stage,
                    operation=name,
                    gate="mcp_transport_health",
                    result="PASS",
                    reason="fresh MCP worker session recovered verifier transport",
                    details={
                        "prior_failures": tuple(failures),
                        "consecutive_failure_count": 0,
                    },
                )
            return value

    setattr(execute, _MCP_MARKER, True)
    execute.__wrapped__ = current  # type: ignore[attr-defined]
    pool_class._execute = execute


def _install_parallel_external_provider(external_mcp_router_module: Any) -> None:
    router_class = external_mcp_router_module.ExternalMCPRouter
    current = router_class._call_provider
    if bool(getattr(current, _EXTERNAL_MARKER, False)):
        return

    def call_provider(
        self: Any,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self._call_provider_async(
                server_name,
                entry,
                tool=tool,
                arguments=arguments,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(run)

        value: dict[str, Any] = {}
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                value["result"] = anyio.run(run)
            except BaseException as exc:  # pragma: no cover - event-loop bridge
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout_seconds + 5.0)
        if thread.is_alive():
            raise external_mcp_router_module.ExternalMCPError(
                f"External MCP {server_name} exceeded the synchronous bridge timeout."
            )
        if errors:
            raise external_mcp_router_module.ExternalMCPError(str(errors[0])) from errors[0]
        return value["result"]

    setattr(call_provider, _EXTERNAL_MARKER, True)
    call_provider.__wrapped__ = current  # type: ignore[attr-defined]
    router_class._call_provider = call_provider


def _is_sqlite_file(target: Path) -> bool:
    """Recognize an existing SQLite index without importing another module's private helper."""

    try:
        with target.open("rb") as input_file:
            return input_file.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
    except OSError:
        return False


def _lsh_table_ready(module: Any, connection: sqlite3.Connection) -> bool:
    if not module.table_exists(connection, "mmm_semantic_lsh"):
        return False
    if not module.table_exists(connection, _LSH_STATE_TABLE):
        return False
    row = connection.execute(
        f"SELECT value FROM {_LSH_STATE_TABLE} WHERE key = ?",
        (_LSH_STATE_KEY,),
    ).fetchone()
    return bool(row and str(row[0]) == _LSH_STATE_VERSION)


def _publish_lsh_ready(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_LSH_STATE_TABLE} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"INSERT OR REPLACE INTO {_LSH_STATE_TABLE}(key, value) VALUES (?, ?)",
        (_LSH_STATE_KEY, _LSH_STATE_VERSION),
    )
    connection.commit()


def _invalidate_lsh_ready(module: Any, target: Path) -> None:
    if not target.is_file() or not _is_sqlite_file(target):
        return
    try:
        with sqlite3.connect(str(target), timeout=30.0) as connection:
            if module.table_exists(connection, _LSH_STATE_TABLE):
                connection.execute(
                    f"DELETE FROM {_LSH_STATE_TABLE} WHERE key = ?",
                    (_LSH_STATE_KEY,),
                )
                connection.commit()
    except sqlite3.Error as exc:
        raise RuntimeError("cannot invalidate semantic LSH ready marker") from exc


def _install_rag_lsh_ready_contract(research_rag_performance_module: Any) -> None:
    module = research_rag_performance_module
    if bool(getattr(module._lsh_candidate_rows, _RAG_MARKER, False)):
        return

    original_ensure = module._ensure_semantic_lsh
    original_build = module.build_index

    def ensure_semantic_lsh(connection: sqlite3.Connection) -> None:
        original_ensure(connection)
        _publish_lsh_ready(connection)

    ensure_semantic_lsh._mmm_no_blanket_delete_v1 = True  # type: ignore[attr-defined]
    ensure_semantic_lsh._mmm_lsh_ready_publisher_v1 = True  # type: ignore[attr-defined]
    ensure_semantic_lsh.__wrapped__ = original_ensure  # type: ignore[attr-defined]

    def build_index(
        index: Any,
        roots: Sequence[str | Path],
        *,
        metadata: dict[str, Any],
        router: Any | None = None,
        semantic: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        target = Path(index.index_path).expanduser().resolve()
        if semantic:
            _invalidate_lsh_ready(module, target)
        result = original_build(
            index,
            roots,
            metadata=metadata,
            router=router,
            semantic=semantic,
            max_files=max_files,
        )
        if semantic and target.is_file() and _is_sqlite_file(target):
            try:
                with sqlite3.connect(str(target), timeout=30.0) as connection:
                    if not _lsh_table_ready(module, connection):
                        ensure_semantic_lsh(connection)
            except Exception:
                pass
        return result

    def lsh_candidate_rows(
        connection: sqlite3.Connection,
        query_vector: Sequence[float],
        *,
        target: int,
        cap: int,
    ) -> list[sqlite3.Row]:
        if not _lsh_table_ready(module, connection):
            raise RuntimeError("semantic LSH side index is not ready")
        signatures = module._signatures([query_vector])
        if not signatures:
            return []
        sig_a, sig_b = signatures[0]

        def query(radius: int) -> list[sqlite3.Row]:
            a = module._hamming_neighborhood(sig_a, module._LSH_BITS, radius)
            b = module._hamming_neighborhood(sig_b, module._LSH_BITS, radius)
            placeholders_a = ",".join("?" for _ in a)
            placeholders_b = ",".join("?" for _ in b)
            return connection.execute(
                f"""
                SELECT c.chunk_id, c.source_path, c.text, c.start_line, c.end_line,
                       c.sha256, c.embedding
                FROM mmm_semantic_lsh AS l
                JOIN chunks AS c ON c.chunk_id = l.chunk_id
                WHERE l.sig_a IN ({placeholders_a}) OR l.sig_b IN ({placeholders_b})
                ORDER BY c.source_path, c.start_line, c.chunk_id
                LIMIT ?
                """,
                (*a, *b, cap),
            ).fetchall()

        rows = query(1)
        if len(rows) < target:
            rows = query(2)
        return rows[:cap]

    setattr(lsh_candidate_rows, _RAG_MARKER, True)
    lsh_candidate_rows.__wrapped__ = module._lsh_candidate_rows  # type: ignore[attr-defined]
    build_index._mmm_lsh_build_invalidation_v1 = True  # type: ignore[attr-defined]
    build_index.__wrapped__ = original_build  # type: ignore[attr-defined]
    module._ensure_semantic_lsh = ensure_semantic_lsh
    module._lsh_candidate_rows = lsh_candidate_rows
    module.build_index = build_index


def install(
    *,
    mcp_transport_pool_module: Any,
    external_mcp_router_module: Any,
    research_rag_performance_module: Any,
) -> None:
    _install_nonblocking_transport(mcp_transport_pool_module)
    _install_parallel_external_provider(external_mcp_router_module)
    _install_rag_lsh_ready_contract(research_rag_performance_module)


def assert_installed(
    *,
    mcp_transport_pool_module: Any,
    external_mcp_router_module: Any,
    research_rag_performance_module: Any,
) -> None:
    if getattr(mcp_transport_pool_module.MCPTransportPool._execute, _MCP_MARKER, False) is not True:
        raise RuntimeError("non-blocking MCP transport execute contract is not installed")
    if getattr(external_mcp_router_module.ExternalMCPRouter._call_provider, _EXTERNAL_MARKER, False) is not True:
        raise RuntimeError("parallel external MCP provider contract is not installed")
    if getattr(research_rag_performance_module._lsh_candidate_rows, _RAG_MARKER, False) is not True:
        raise RuntimeError("RAG LSH query-ready contract is not installed")


__all__ = [
    "assert_installed",
    "install",
    "_MCP_MAX_CONSECUTIVE_INFRA_FAILURES",
    "_is_transient_mcp_transport_failure",
]
