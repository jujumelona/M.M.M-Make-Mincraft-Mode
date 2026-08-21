from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import threading
import weakref
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Callable, Mapping

import anyio


_MIN_WORKERS = 1
_MAX_WORKERS = 16
_DEFAULT_WORKERS = 4
_DEFAULT_PENDING_PER_WORKER = 32
_PATCH_LOCK = threading.Lock()
_PATCH_INSTALLED = False

SessionFactory = Callable[
    [str, Mapping[str, str], float],
    AsyncContextManager[Any],
]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _tool_schema_fingerprint(listed: Any) -> str:
    rows: list[dict[str, Any]] = []
    for item in getattr(listed, "tools", ()) or ():
        schema = getattr(item, "inputSchema", None)
        if schema is None:
            schema = getattr(item, "input_schema", None)
        rows.append(
            {
                "name": str(getattr(item, "name", "")).strip(),
                "description": str(getattr(item, "description", "") or ""),
                "input_schema": _jsonable(schema),
            }
        )
    rows.sort(key=lambda item: item["name"])
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _schema_scope_key(
    stage: str,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[str, str, float]:
    env_payload = json.dumps(
        sorted((str(key), str(value)) for key, value in env.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        stage,
        hashlib.sha256(env_payload).hexdigest(),
        float(timeout_seconds),
    )


@dataclass(frozen=True)
class _TransportRequest:
    operation: str
    stage: str
    env: Mapping[str, str]
    timeout_seconds: float
    result: concurrent.futures.Future[Any]
    name: str = ""
    arguments: Mapping[str, Any] | None = None
    expected_schema_sha256: str = ""


@asynccontextmanager
async def _stdio_session(
    stage: str,
    env: Mapping[str, str],
    timeout_seconds: float,
):
    """Open one reusable stdio transport with LIFO-safe AnyIO scope ownership."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:  # pragma: no cover - dependency failure
        from .agent_tool_runtime import AgentToolRuntimeError

        raise AgentToolRuntimeError(
            "The pinned MCP Python client is unavailable"
        ) from exc

    stack = AsyncExitStack()
    try:
        errlog = stack.enter_context(
            tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "minecraft_mod_ai.mcp_server"],
            env=dict(env),
        )
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params, errlog=errlog)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        with anyio.fail_after(timeout_seconds):
            await session.initialize()
        yield session
    finally:
        await stack.aclose()


class _SessionWorker:
    """Own one event loop and at most one live MCP subprocess/session."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        max_pending: int,
        name: str,
    ) -> None:
        self._session_factory = session_factory
        self._max_pending = max_pending
        self._name = name
        self._state_lock = threading.Lock()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_TransportRequest | None] | None = None
        self._startup_error: BaseException | None = None
        self._closed = False
        self._pending = 0

    @property
    def pending(self) -> int:
        with self._state_lock:
            return self._pending

    def reserve(self) -> None:
        """Reserve one queue position while the pool still owns dispatch ordering."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("MCP transport worker is closed")
            self._pending += 1

    def submit(self, request: _TransportRequest) -> None:
        request.result.add_done_callback(lambda _future: self._release_pending())
        try:
            self._ensure_started()
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("MCP transport worker is closed")
                startup_error = self._startup_error
                loop = self._loop
                queue = self._queue
            if startup_error is not None:
                raise RuntimeError("MCP transport worker failed to start") from startup_error
            if loop is None or queue is None:
                raise RuntimeError("MCP transport worker did not initialize")

            enqueue = asyncio.run_coroutine_threadsafe(queue.put(request), loop)
            enqueue.result(timeout=request.timeout_seconds)
        except BaseException:
            if not request.result.done():
                request.result.cancel()
            raise

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            loop = self._loop
            queue = self._queue
        if thread is None:
            return
        if loop is not None and queue is not None and thread.is_alive():
            try:
                enqueue = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                enqueue.result(timeout=5.0)
            except BaseException:
                pass
        if thread is not threading.current_thread():
            thread.join(timeout=10.0)

    def _ensure_started(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("MCP transport worker is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._thread_main,
                    name=self._name,
                    daemon=True,
                )
                self._thread.start()
        self._ready.wait(timeout=5.0)
        if not self._ready.is_set():
            raise RuntimeError("MCP transport worker startup timed out")

    def _thread_main(self) -> None:
        try:
            anyio.run(self._main)
        except BaseException as exc:  # pragma: no cover - catastrophic loop failure
            with self._state_lock:
                self._startup_error = exc
            self._fail_queued(exc)
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_pending)
        self._ready.set()

        context: AsyncContextManager[Any] | None = None
        session: Any = None
        current_stage = ""
        current_env: dict[str, str] | None = None
        current_timeout = 0.0
        current_listed_tools: Any = None
        current_schema_sha256 = ""

        try:
            while True:
                request = await self._queue.get()
                if request is None:
                    return
                # A caller can cancel after enqueue while this worker is still busy.
                # Never turn an already-cancelled queued request into a late write or
                # other side effect that the caller has been told will not complete.
                if request.result.cancelled():
                    continue
                try:
                    requested_env = dict(request.env)
                    opened_now = False
                    if (
                        session is None
                        or current_stage != request.stage
                        or current_env != requested_env
                        or current_timeout != request.timeout_seconds
                    ):
                        if context is not None:
                            await context.__aexit__(None, None, None)
                        context = self._session_factory(
                            request.stage,
                            requested_env,
                            request.timeout_seconds,
                        )
                        session = await context.__aenter__()
                        current_stage = request.stage
                        current_env = requested_env
                        current_timeout = request.timeout_seconds
                        with anyio.fail_after(request.timeout_seconds):
                            current_listed_tools = await session.list_tools()
                        current_schema_sha256 = _tool_schema_fingerprint(
                            current_listed_tools
                        )
                        opened_now = True

                    with anyio.fail_after(request.timeout_seconds):
                        if request.operation == "list_tools":
                            if not opened_now:
                                current_listed_tools = await session.list_tools()
                                current_schema_sha256 = _tool_schema_fingerprint(
                                    current_listed_tools
                                )
                            value = current_listed_tools
                        elif request.operation == "call_tool":
                            expected = request.expected_schema_sha256
                            if expected and current_schema_sha256 != expected:
                                raise RuntimeError(
                                    "MCP pooled worker schema drift detected before execution: "
                                    f"expected={expected} actual={current_schema_sha256}"
                                )
                            value = await session.call_tool(
                                request.name,
                                arguments=dict(request.arguments or {}),
                            )
                        else:  # pragma: no cover - internal invariant
                            raise RuntimeError(
                                f"Unknown MCP transport operation: {request.operation}"
                            )
                    if not request.result.done():
                        request.result.set_result(value)
                except BaseException as exc:
                    if not request.result.done():
                        request.result.set_exception(exc)
                    if context is not None:
                        try:
                            await context.__aexit__(
                                type(exc),
                                exc,
                                exc.__traceback__,
                            )
                        except BaseException:
                            pass
                    context = None
                    session = None
                    current_stage = ""
                    current_env = None
                    current_timeout = 0.0
                    current_listed_tools = None
                    current_schema_sha256 = ""
        finally:
            if context is not None:
                try:
                    await context.__aexit__(None, None, None)
                except BaseException:
                    pass

    def _fail_queued(self, exc: BaseException) -> None:
        queue = self._queue
        if queue is None:
            return
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if request is not None and not request.result.done():
                request.result.set_exception(exc)

    def _release_pending(self) -> None:
        with self._state_lock:
            self._pending = max(0, self._pending - 1)


class MCPTransportPool:
    """Bounded reusable MCP transport pool for one AgentToolRuntime.

    Worker slots are global to the runtime rather than multiplied by stage. A slot
    keeps its stdio subprocess warm for repeated calls and swaps sessions only when
    stage/environment/timeout changes. ModelRouter's existing read-wave parallelism
    therefore stays bounded while process startup/initialize is amortized.

    Every scope also owns one canonical raw tools/list fingerprint. A worker that
    restarts with a different tool surface is rejected before call_tool execution,
    preventing a cached schema from worker A from being used against worker B.
    """

    def __init__(
        self,
        *,
        worker_count: int | None = None,
        max_pending_per_worker: int = _DEFAULT_PENDING_PER_WORKER,
        session_factory: SessionFactory = _stdio_session,
    ) -> None:
        if worker_count is None:
            worker_count = _configured_worker_count()
        worker_count = int(worker_count)
        max_pending_per_worker = int(max_pending_per_worker)
        if not _MIN_WORKERS <= worker_count <= _MAX_WORKERS:
            raise ValueError(
                f"worker_count must be between {_MIN_WORKERS} and {_MAX_WORKERS}"
            )
        if max_pending_per_worker < 1:
            raise ValueError("max_pending_per_worker must be positive")
        self._closed = False
        self._dispatch_lock = threading.Lock()
        self._schema_lock = threading.Lock()
        self._schema_fingerprints: dict[tuple[str, str, float], str] = {}
        self._workers = tuple(
            _SessionWorker(
                session_factory=session_factory,
                max_pending=max_pending_per_worker,
                name=f"mmm_mcp_transport_{index}",
            )
            for index in range(worker_count)
        )

    async def list_tools(
        self,
        *,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> Any:
        listed = await self._execute(
            operation="list_tools",
            stage=stage,
            env=env,
            timeout_seconds=timeout_seconds,
        )
        self._register_schema(
            stage=stage,
            env=env,
            timeout_seconds=timeout_seconds,
            listed=listed,
        )
        return listed

    async def call_tool(
        self,
        *,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
        name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        expected = self._schema_fingerprint_for_scope(
            stage=stage,
            env=env,
            timeout_seconds=timeout_seconds,
        )
        if not expected:
            await self.list_tools(
                stage=stage,
                env=env,
                timeout_seconds=timeout_seconds,
            )
            expected = self._schema_fingerprint_for_scope(
                stage=stage,
                env=env,
                timeout_seconds=timeout_seconds,
            )
        if not expected:  # pragma: no cover - defensive invariant
            raise RuntimeError("MCP schema fingerprint was not established before execution")
        return await self._execute(
            operation="call_tool",
            stage=stage,
            env=env,
            timeout_seconds=timeout_seconds,
            name=name,
            arguments=arguments,
            expected_schema_sha256=expected,
        )

    def close(self) -> None:
        with self._dispatch_lock:
            if self._closed:
                return
            self._closed = True
            workers = self._workers
        with self._schema_lock:
            self._schema_fingerprints.clear()
        for worker in workers:
            worker.close()

    def _register_schema(
        self,
        *,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
        listed: Any,
    ) -> str:
        key = _schema_scope_key(stage, env, timeout_seconds)
        fingerprint = _tool_schema_fingerprint(listed)
        with self._schema_lock:
            previous = self._schema_fingerprints.get(key)
            if previous is None:
                self._schema_fingerprints[key] = fingerprint
                return fingerprint
            if previous != fingerprint:
                raise RuntimeError(
                    "MCP tools/list schema drift detected across pooled workers: "
                    f"expected={previous} actual={fingerprint}"
                )
            return previous

    def _schema_fingerprint_for_scope(
        self,
        *,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> str:
        key = _schema_scope_key(stage, env, timeout_seconds)
        with self._schema_lock:
            return self._schema_fingerprints.get(key, "")

    async def _execute(
        self,
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
        request = _TransportRequest(
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
        worker.submit(request)
        return await asyncio.wrap_future(future)

    def _reserve_worker(self) -> _SessionWorker:
        with self._dispatch_lock:
            if self._closed:
                raise RuntimeError("MCP transport pool is closed")
            worker = min(self._workers, key=lambda candidate: candidate.pending)
            worker.reserve()
            return worker


class _PooledSession:
    """Async session facade matching the interface AgentToolRuntime already uses."""

    def __init__(
        self,
        pool: MCPTransportPool,
        *,
        stage: str,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> None:
        self._pool = pool
        self._stage = stage
        self._env = dict(env)
        self._timeout_seconds = float(timeout_seconds)

    async def __aenter__(self) -> "_PooledSession":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def list_tools(self) -> Any:
        return await self._pool.list_tools(
            stage=self._stage,
            env=self._env,
            timeout_seconds=self._timeout_seconds,
        )

    async def call_tool(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
    ) -> Any:
        return await self._pool.call_tool(
            stage=self._stage,
            env=self._env,
            timeout_seconds=self._timeout_seconds,
            name=name,
            arguments=arguments,
        )


def install_agent_mcp_transport_pool() -> None:
    """Install bounded session reuse without changing AgentToolRuntime's public API."""
    global _PATCH_INSTALLED

    from .agent_tool_runtime import AgentToolRuntime

    with _PATCH_LOCK:
        if _PATCH_INSTALLED:
            return

        def pooled_session(self: Any, stage: str) -> _PooledSession:
            pool = getattr(self, "_mcp_transport_pool", None)
            if pool is None:
                with self._lock:
                    pool = getattr(self, "_mcp_transport_pool", None)
                    if pool is None:
                        pool = MCPTransportPool()
                        self._mcp_transport_pool = pool
                        self._mcp_transport_pool_finalizer = weakref.finalize(
                            self,
                            pool.close,
                        )
            return _PooledSession(
                pool,
                stage=stage,
                env=self._child_env(stage),
                timeout_seconds=self.timeout_seconds,
            )

        AgentToolRuntime._session = pooled_session
        _PATCH_INSTALLED = True


def _configured_worker_count() -> int:
    raw = os.environ.get(
        "MMM_MCP_SESSION_WORKERS",
        os.environ.get("MMM_AGENT_PARALLEL_READS", str(_DEFAULT_WORKERS)),
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_WORKERS
    return max(_MIN_WORKERS, min(value, _MAX_WORKERS))