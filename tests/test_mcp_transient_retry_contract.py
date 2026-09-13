from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntimeError
from minecraft_mod_ai.mcp_transient_retry_contract import (
    _MAX_CONSECUTIVE_INFRA_FAILURES,
    _call_with_transient_retry,
    _is_transient_transport_failure,
)


class _Pool:
    def __init__(self, *, cleanup_error: BaseException | None = None) -> None:
        self.closed = 0
        self.cleanup_error = cleanup_error

    def close(self) -> None:
        self.closed += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _Finalizer:
    alive = True

    def __init__(self) -> None:
        self.detached = 0

    def detach(self) -> None:
        self.detached += 1
        self.alive = False


class _Runtime:
    def __init__(self, pool: _Pool | None = None) -> None:
        self._lock = threading.RLock()
        self._mcp_transport_pool = pool
        self._mcp_transport_pool_finalizer = _Finalizer() if pool is not None else None


def test_timeout_retries_same_verifier_after_invalidating_pool() -> None:
    pool = _Pool()
    runtime = _Runtime(pool)
    calls = 0

    async def operation(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("MCP call timed out")
        assert runtime._mcp_transport_pool is None
        return {"diagnostics": []}

    result = asyncio.run(
        _call_with_transient_retry(
            runtime,
            operation,
            "generation",
            "java_diagnostics",
            {},
        )
    )

    assert result == {"diagnostics": []}
    assert calls == 2
    assert pool.closed == 1


def test_three_consecutive_transport_failures_exhaust_retry_budget() -> None:
    runtime = _Runtime(_Pool())
    calls = 0

    async def operation(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise ConnectionResetError("MCP channel closed")

    with pytest.raises(AgentToolRuntimeError, match="MCP_TRANSIENT_RETRY_EXHAUSTED") as raised:
        asyncio.run(
            _call_with_transient_retry(
                runtime,
                operation,
                "generation",
                "java_diagnostics",
                {},
            )
        )

    assert calls == _MAX_CONSECUTIVE_INFRA_FAILURES
    assert "retry_history=" in str(raised.value)
    assert isinstance(raised.value.__cause__, ConnectionResetError)


def test_success_resets_failure_streak_for_next_verifier_call() -> None:
    runtime = _Runtime(_Pool())
    first_calls = 0

    async def first(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal first_calls
        first_calls += 1
        if first_calls < _MAX_CONSECUTIVE_INFRA_FAILURES:
            runtime._mcp_transport_pool = _Pool()
            raise TimeoutError("temporary timeout")
        return {"diagnostics": []}

    assert asyncio.run(
        _call_with_transient_retry(
            runtime,
            first,
            "generation",
            "java_diagnostics",
            {},
        )
    ) == {"diagnostics": []}

    second_calls = 0

    async def second(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal second_calls
        second_calls += 1
        if second_calls == 1:
            runtime._mcp_transport_pool = _Pool()
            raise TimeoutError("new timeout")
        return {"diagnostics": []}

    assert asyncio.run(
        _call_with_transient_retry(
            runtime,
            second,
            "generation",
            "java_diagnostics",
            {},
        )
    ) == {"diagnostics": []}
    assert second_calls == 2


def test_deterministic_invalid_argument_is_not_retried() -> None:
    pool = _Pool()
    runtime = _Runtime(pool)
    calls = 0

    async def operation(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AgentToolRuntimeError("Invalid task diagnostic path: '../Escape.java'")

    with pytest.raises(AgentToolRuntimeError, match="Invalid task diagnostic path"):
        asyncio.run(
            _call_with_transient_retry(
                runtime,
                operation,
                "generation",
                "java_diagnostics",
                {},
            )
        )

    assert calls == 1
    assert pool.closed == 0


def test_cleanup_failure_never_masks_primary_timeout() -> None:
    pool = _Pool(cleanup_error=RuntimeError("no running event loop"))
    runtime = _Runtime(pool)
    calls = 0

    async def operation(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise TimeoutError("primary MCP timeout")

    with pytest.raises(AgentToolRuntimeError) as raised:
        asyncio.run(
            _call_with_transient_retry(
                runtime,
                operation,
                "generation",
                "java_diagnostics",
                {},
            )
        )

    assert calls == _MAX_CONSECUTIVE_INFRA_FAILURES
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert "primary MCP timeout" in str(raised.value)
    assert "no running event loop" in str(raised.value)


def test_mutation_tool_is_never_transport_retried() -> None:
    pool = _Pool()
    runtime = _Runtime(pool)
    calls = 0

    async def operation(
        _runtime: Any,
        _stage: str,
        _name: str,
        _arguments: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise TimeoutError("write outcome unknown")

    with pytest.raises(TimeoutError, match="write outcome unknown"):
        asyncio.run(
            _call_with_transient_retry(
                runtime,
                operation,
                "generation",
                "apply_source_patch",
                {},
            )
        )

    assert calls == 1
    assert pool.closed == 0


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timed out"),
        ConnectionResetError("reset"),
        EOFError("EOF"),
        RuntimeError("MCP channel closed"),
    ],
)
def test_transient_classifier_accepts_transport_failures(exc: BaseException) -> None:
    assert _is_transient_transport_failure(exc)


@pytest.mark.parametrize(
    "exc",
    [
        AgentToolRuntimeError("Invalid task diagnostic path"),
        AgentToolRuntimeError("schema drift detected"),
        asyncio.CancelledError(),
    ],
)
def test_transient_classifier_rejects_deterministic_or_cancelled_failures(
    exc: BaseException,
) -> None:
    assert not _is_transient_transport_failure(exc)
