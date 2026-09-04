from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from typing import Any

import anyio

from .root_cause_trace import emit_root_cause, exception_chain

_INSTALL_LOCK = threading.Lock()
_INSTALL_ATTR = "_mmm_child_trace_contract_installed"


def _probe_fileno(stream: Any) -> tuple[int | None, str | None]:
    """Return a usable OS descriptor without assuming notebook streams expose one."""

    if stream is None:
        return None, "stream is None"
    try:
        descriptor = stream.fileno()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(descriptor, int) or descriptor < 0:
        return None, f"invalid fileno: {descriptor!r}"
    return descriptor, None


@contextmanager
def _subprocess_stderr_target() -> Iterator[tuple[Any, str, dict[str, str]]]:
    """Choose a parent-visible stderr object that subprocess.Popen can actually use.

    IPython/Colab replace ``sys.stderr`` with an OutStream whose ``fileno()`` raises
    ``io.UnsupportedOperation``. Passing that object to MCP's stdio client therefore
    fails before the child process exists. Prefer the active stderr when it is a real
    fd-backed stream, then the interpreter's original stderr, and finally a duplicate
    of process fd 2. The fd-2 duplicate keeps child/JDT diagnostics parent-visible and
    avoids the old TemporaryFile behavior that swallowed the first root cause.
    """

    probe_failures: dict[str, str] = {}
    candidates = (
        ("parent_stderr", sys.stderr),
        ("parent_dunder_stderr", getattr(sys, "__stderr__", None)),
    )
    for route, stream in candidates:
        _descriptor, failure = _probe_fileno(stream)
        if failure is None:
            yield stream, route, probe_failures
            return
        probe_failures[route] = failure

    try:
        duplicate_fd = os.dup(2)
        fallback = os.fdopen(duplicate_fd, "wb", buffering=0, closefd=True)
    except Exception as exc:
        detail = "; ".join(
            f"{route}={reason}" for route, reason in probe_failures.items()
        )
        raise RuntimeError(
            "No subprocess-compatible parent stderr is available"
            + (f" ({detail})" if detail else "")
        ) from exc

    try:
        yield fallback, "parent_fd2_duplicate", probe_failures
    finally:
        fallback.close()


@asynccontextmanager
async def traced_stdio_session(
    stage: str,
    env: Mapping[str, str],
    timeout_seconds: float,
):
    """Open MCP stdio while preserving child stderr in the parent one-run log.

    MCP JSON-RPC remains exclusively on stdout. stderr is intentionally inherited by
    the parent so root-cause events emitted by the child MCP server and JDT lifecycle
    are not captured in a temporary file and silently discarded.

    Per-tool START/PASS/FAIL events are emitted by the child stage-tool wrapper itself.
    Transport/initialize/teardown failures are emitted here. This deliberately avoids
    another runtime method rebind, preserving the reviewed mutation surface.
    """

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:  # pragma: no cover - dependency failure
        from .agent_tool_runtime import AgentToolRuntimeError

        emit_root_cause(
            "mcp_transport_dependency_failure",
            stage=stage,
            operation="mcp_stdio_session",
            gate="mcp_client_import",
            result="FAIL",
            reason=f"{type(exc).__name__}: {exc}",
            details={"exception_chain": exception_chain(exc)},
            exc=exc,
        )
        raise AgentToolRuntimeError(
            "The pinned MCP Python client is unavailable"
        ) from exc

    started = time.monotonic()
    stack = AsyncExitStack()
    try:
        stderr_context = _subprocess_stderr_target()
        stderr_target, stderr_route, stderr_probe_failures = stderr_context.__enter__()
    except BaseException as exc:
        emit_root_cause(
            "mcp_transport_stderr_route_failure",
            stage=stage,
            operation="mcp_stdio_session",
            gate="stderr_route",
            result="FAIL",
            reason=f"{type(exc).__name__}: {exc}",
            details={"exception_chain": exception_chain(exc)},
            exc=exc,
        )
        raise

    emit_root_cause(
        "mcp_transport_session_start",
        stage=stage,
        operation="mcp_stdio_session",
        gate="transport_start",
        result="START",
        details={
            "command": sys.executable,
            "module": "minecraft_mod_ai.mcp_server",
            "timeout_seconds": float(timeout_seconds),
            "child_env_keys": sorted(str(key) for key in env),
            "stderr_route": stderr_route,
            "stderr_probe_failures": stderr_probe_failures,
            "stderr_stream_type": type(stderr_target).__name__,
        },
    )
    close_error: BaseException | None = None
    active_error: BaseException | None = None
    try:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "minecraft_mod_ai.mcp_server"],
            env=dict(env),
        )
        # Keep child/JDT root-cause events parent-visible, but only pass a stream
        # whose fileno() is valid for subprocess.Popen (not IPython OutStream).
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params, errlog=stderr_target)
        )
        emit_root_cause(
            "mcp_transport_process_ready",
            stage=stage,
            operation="mcp_stdio_session",
            gate="process_spawn",
            result="PASS",
            details={
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "stderr_route": stderr_route,
            },
        )
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        initialize_started = time.monotonic()
        with anyio.fail_after(timeout_seconds):
            initialize_result = await session.initialize()
        emit_root_cause(
            "mcp_transport_initialized",
            stage=stage,
            operation="mcp_stdio_session",
            gate="initialize",
            result="PASS",
            details={
                "initialize_elapsed_ms": round(
                    (time.monotonic() - initialize_started) * 1000.0,
                    3,
                ),
                "result_type": type(initialize_result).__name__,
            },
        )
        yield session
    except BaseException as exc:
        active_error = exc
        emit_root_cause(
            "mcp_transport_session_failure",
            stage=stage,
            operation="mcp_stdio_session",
            gate="transport_or_initialize",
            result="FAIL",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "exception_chain": exception_chain(exc),
                "stderr_route": stderr_route,
                "stderr_probe_failures": stderr_probe_failures,
            },
            exc=exc,
        )
        raise
    finally:
        try:
            await stack.aclose()
        except BaseException as exc:  # pragma: no cover - teardown failure
            close_error = exc
            emit_root_cause(
                "mcp_transport_close_failure",
                stage=stage,
                operation="mcp_stdio_session",
                gate="transport_close",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={"exception_chain": exception_chain(exc)},
                exc=exc,
            )
            if active_error is None:
                raise
        finally:
            try:
                stderr_context.__exit__(None, None, None)
            finally:
                emit_root_cause(
                    "mcp_transport_session_closed",
                    stage=stage,
                    operation="mcp_stdio_session",
                    gate="transport_close",
                    result="FAIL" if close_error is not None else "PASS",
                    reason=(
                        f"{type(close_error).__name__}: {close_error}"
                        if close_error is not None
                        else "MCP stdio session closed"
                    ),
                    details={
                        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                        "had_active_error": active_error is not None,
                        "stderr_route": stderr_route,
                        "stderr_probe_failures": stderr_probe_failures,
                    },
                )


def install(mcp_transport_pool_module: Any) -> None:
    """Install parent-visible child stderr before the runtime pool is materialized."""

    with _INSTALL_LOCK:
        if bool(getattr(mcp_transport_pool_module, _INSTALL_ATTR, False)):
            return

        pool_cls = mcp_transport_pool_module.MCPTransportPool
        original_stdio_session = mcp_transport_pool_module._stdio_session
        kwdefaults = dict(pool_cls.__init__.__kwdefaults__ or {})
        current_default = kwdefaults.get("session_factory")
        if current_default is not original_stdio_session:
            raise RuntimeError(
                "MCP child trace contract expected the baseline _stdio_session default; "
                f"found {current_default!r}"
            )
        kwdefaults["session_factory"] = traced_stdio_session
        pool_cls.__init__.__kwdefaults__ = kwdefaults
        mcp_transport_pool_module._stdio_session = traced_stdio_session
        setattr(mcp_transport_pool_module, _INSTALL_ATTR, True)


__all__ = ["install", "traced_stdio_session"]
