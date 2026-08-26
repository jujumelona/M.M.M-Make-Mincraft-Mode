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
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

_MCP_MARKER = "_mmm_nonblocking_transport_execute_v1"
_EXTERNAL_MARKER = "_mmm_external_mcp_parallel_provider_v1"
_RAG_MARKER = "_mmm_rag_lsh_query_ready_v1"
_LSH_STATE_TABLE = "mmm_semantic_lsh_state"
_LSH_STATE_KEY = "ready"
_LSH_STATE_VERSION = "v1"
_SQLITE_MAGIC = b"SQLite format 3\x00"


async def _submit_without_blocking_loop(worker: Any, request: Any) -> None:
    """Apply bounded queue backpressure without blocking the caller event loop."""

    request.result.add_done_callback(lambda _future: worker._release_pending())
    enqueue: concurrent.futures.Future[Any] | None = None
    try:
        # Worker startup can wait on a threading.Event for up to five seconds. Keep
        # that wait outside the model/tool event loop.
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
        await _submit_without_blocking_loop(worker, request)
        return await asyncio.wrap_future(future)

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

        # Every provider call owns an independent MCP session. The old global router
        # RLock therefore serialized unrelated providers without protecting shared
        # transport state. Keep only the existing sync/async bridge semantics.
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
    # Also commits stale-row cleanup performed by the canonical reconciler when no
    # missing embedding batch happened to trigger its internal commit loop.
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
        # Never mutate a semantic index while an old ready marker may still be
        # visible. A failed invalidation therefore blocks the build rather than
        # permitting stale side-index candidates during concurrent search.
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
                # Side index is optimization-only. Leaving the marker absent forces
                # semantic search back to the canonical exhaustive path.
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


__all__ = ["assert_installed", "install"]