from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .root_cause_trace import emit_root_cause

_MAX_CONSECUTIVE_INFRA_FAILURES = 3
_RETRY_SAFE_VERIFIERS = frozenset(
    {
        "java_diagnostics",
        "jdt_diagnostics",
        "run_gradle_build",
        "gradle_build",
        "run_gametest",
    }
)
_DETERMINISTIC_MARKERS = (
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
_TRANSIENT_MARKERS = (
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


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
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


def _is_transient_transport_failure(exc: BaseException) -> bool:
    chain = _exception_chain(exc)
    if any(isinstance(item, asyncio.CancelledError) for item in chain):
        return False

    text = " | ".join(f"{type(item).__name__}: {item}" for item in chain).casefold()
    if any(marker in text for marker in _DETERMINISTIC_MARKERS):
        return False

    if any(
        isinstance(item, (TimeoutError, ConnectionError, EOFError, BrokenPipeError))
        for item in chain
    ):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _invalidate_runtime_transport(runtime: Any) -> str | None:
    """Detach and close the current pool; cleanup failures stay secondary."""
    lock = getattr(runtime, "_lock", None)
    if lock is None:
        lock = threading.RLock()

    with lock:
        pool = getattr(runtime, "_mcp_transport_pool", None)
        if pool is None:
            return None
        runtime._mcp_transport_pool = None
        finalizer = getattr(runtime, "_mcp_transport_pool_finalizer", None)
        runtime._mcp_transport_pool_finalizer = None

    if finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer.detach()
        except BaseException:
            pass

    try:
        pool.close()
    except BaseException as cleanup_exc:  # cleanup must never replace primary timeout
        return f"{type(cleanup_exc).__name__}: {cleanup_exc}"
    return None


async def _call_with_transient_retry(
    runtime: Any,
    original: Callable[[Any, str, str, Mapping[str, Any]], Awaitable[dict[str, Any]]],
    stage: str,
    name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if name not in _RETRY_SAFE_VERIFIERS:
        return await original(runtime, stage, name, arguments)

    from .agent_tool_runtime import AgentToolRuntimeError

    history: list[dict[str, Any]] = []
    for attempt in range(1, _MAX_CONSECUTIVE_INFRA_FAILURES + 1):
        try:
            result = await original(runtime, stage, name, arguments)
        except BaseException as exc:
            if not _is_transient_transport_failure(exc):
                raise

            cleanup_error = _invalidate_runtime_transport(runtime)
            entry = {
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cleanup_error": cleanup_error,
            }
            history.append(entry)
            exhausted = attempt >= _MAX_CONSECUTIVE_INFRA_FAILURES
            emit_root_cause(
                "mcp_verifier_transport_retry",
                stage=stage,
                operation=name,
                gate="mcp_transport_health",
                result="FAIL" if exhausted else "RETRY",
                reason=(
                    "transient verifier transport retries exhausted"
                    if exhausted
                    else "transient verifier transport failure; fresh session required"
                ),
                details={
                    "attempt": attempt,
                    "max_attempts": _MAX_CONSECUTIVE_INFRA_FAILURES,
                    "retry_history": tuple(history),
                    "transport_invalidated": True,
                },
                exc=exc,
            )
            if exhausted:
                raise AgentToolRuntimeError(
                    "MCP_TRANSIENT_RETRY_EXHAUSTED: verifier transport failed "
                    f"{_MAX_CONSECUTIVE_INFRA_FAILURES} consecutive times; "
                    f"retry_history={json.dumps(history, ensure_ascii=False, sort_keys=True)}"
                ) from exc

            # pool.close() above joins the old worker. Yield once before the next
            # attempt so the next _session() acquisition cannot observe stale work.
            await asyncio.sleep(0)
            continue

        if history:
            emit_root_cause(
                "mcp_verifier_transport_recovered",
                stage=stage,
                operation=name,
                gate="mcp_transport_health",
                result="PASS",
                reason="fresh MCP session recovered verifier transport",
                details={
                    "attempt": attempt,
                    "prior_failures": tuple(history),
                    "consecutive_failure_count": 0,
                },
            )
        return result

    raise AssertionError("unreachable")


_PATCH_LOCK = threading.Lock()
_PATCH_INSTALLED = False


def install() -> None:
    global _PATCH_INSTALLED
    from .agent_tool_runtime import AgentToolRuntime

    with _PATCH_LOCK:
        if _PATCH_INSTALLED:
            return

        original = AgentToolRuntime._call_tool_async
        if getattr(original, "__mmm_transient_retry_contract__", False):
            _PATCH_INSTALLED = True
            return

        async def resilient_call_tool_async(
            self: Any,
            stage: str,
            name: str,
            arguments: Mapping[str, Any],
        ) -> dict[str, Any]:
            return await _call_with_transient_retry(
                self,
                original,
                stage,
                name,
                arguments,
            )

        resilient_call_tool_async.__mmm_transient_retry_contract__ = True
        resilient_call_tool_async.__wrapped__ = original
        AgentToolRuntime._call_tool_async = resilient_call_tool_async
        _PATCH_INSTALLED = True


__all__ = ["install"]
