from __future__ import annotations

import functools
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import anyio

from .root_cause_trace import emit_root_cause, exception_chain

_INSTALL_LOCK = threading.Lock()
_INSTALL_ATTR = "_mmm_child_trace_contract_installed"


def _result_shape(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "NoneType"}
    result: dict[str, Any] = {"type": type(value).__name__}
    tools = getattr(value, "tools", None)
    if tools is not None:
        try:
            result["tool_count"] = len(tools)
        except TypeError:
            pass
    structured = getattr(value, "structuredContent", None)
    if structured is None:
        structured = getattr(value, "structured_content", None)
    if isinstance(structured, Mapping):
        result["structured_content_keys"] = sorted(str(key) for key in structured)
        result["structured_status"] = structured.get("status")
        diagnostics = structured.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            result["diagnostic_uri_count"] = len(diagnostics)
            result["diagnostic_item_count"] = sum(
                len(group) if isinstance(group, list) else 0
                for group in diagnostics.values()
            )
    result["is_error"] = bool(
        getattr(value, "isError", getattr(value, "is_error", False))
    )
    return result


@asynccontextmanager
async def traced_stdio_session(
    stage: str,
    env: Mapping[str, str],
    timeout_seconds: float,
):
    """Open MCP stdio while preserving child stderr in the parent one-run log.

    MCP JSON-RPC remains exclusively on stdout.  stderr is intentionally inherited by
    the parent so root-cause events emitted by the child MCP server and JDT lifecycle
    are not captured in a temporary file and silently discarded.
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
            "stderr_route": "parent_stderr",
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
        # Do not replace this with a TemporaryFile.  The one-run diagnostic contract
        # requires child MCP/JDT stderr to remain visible to the invoking process.
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params, errlog=sys.stderr)
        )
        emit_root_cause(
            "mcp_transport_process_ready",
            stage=stage,
            operation="mcp_stdio_session",
            gate="process_spawn",
            result="PASS",
            details={
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "stderr_route": "parent_stderr",
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
                "stderr_route": "parent_stderr",
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
                    "stderr_route": "parent_stderr",
                },
            )


def install(mcp_transport_pool_module: Any) -> None:
    """Install one-run MCP transport tracing before the runtime pool is materialized."""

    with _INSTALL_LOCK:
        if bool(getattr(mcp_transport_pool_module, _INSTALL_ATTR, False)):
            return

        pool_cls = mcp_transport_pool_module.MCPTransportPool
        original_stdio_session = mcp_transport_pool_module._stdio_session
        original_execute = pool_cls._execute

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

        @functools.wraps(original_execute)
        async def traced_execute(
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
            started = time.monotonic()
            emit_root_cause(
                "mcp_transport_request_start",
                stage=stage,
                operation=operation,
                gate="transport_request",
                result="START",
                details={
                    "tool_name": name,
                    "timeout_seconds": float(timeout_seconds),
                    "arguments": dict(arguments or {}),
                    "expected_schema_sha256": expected_schema_sha256,
                },
            )
            try:
                value = await original_execute(
                    self,
                    operation=operation,
                    stage=stage,
                    env=env,
                    timeout_seconds=timeout_seconds,
                    name=name,
                    arguments=arguments,
                    expected_schema_sha256=expected_schema_sha256,
                )
            except BaseException as exc:
                emit_root_cause(
                    "mcp_transport_request_failure",
                    stage=stage,
                    operation=operation,
                    gate="transport_request",
                    result="FAIL",
                    reason=f"{type(exc).__name__}: {exc}",
                    details={
                        "tool_name": name,
                        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                        "exception_chain": exception_chain(exc),
                    },
                    exc=exc,
                )
                raise
            emit_root_cause(
                "mcp_transport_request_result",
                stage=stage,
                operation=operation,
                gate="transport_request",
                result="PASS",
                details={
                    "tool_name": name,
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "result_shape": _result_shape(value),
                },
            )
            return value

        pool_cls._execute = traced_execute
        setattr(mcp_transport_pool_module, _INSTALL_ATTR, True)


__all__ = ["install", "traced_stdio_session"]
