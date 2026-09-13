from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import anyio
import pytest

from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntimeError
from minecraft_mod_ai import runtime_hot_path_contract as contract


@dataclass(frozen=True)
class _Request:
    operation: str
    stage: str
    env: Mapping[str, str]
    timeout_seconds: float
    result: concurrent.futures.Future[Any]
    name: str = ""
    arguments: Mapping[str, Any] | None = None
    expected_schema_sha256: str = ""


def _installed_fake_pool():
    class Pool:
        async def _execute(self, **_kwargs):
            raise AssertionError("baseline execute should be replaced")

        def _reserve_worker(self):
            return object()

    module = SimpleNamespace(MCPTransportPool=Pool, _TransportRequest=_Request)
    contract._install_nonblocking_transport(module)
    return Pool


def test_transient_java_diagnostics_failure_retries_and_recovers(monkeypatch, capsys) -> None:
    pool_cls = _installed_fake_pool()
    attempts = 0

    async def submit(_worker, request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request.result.set_exception(TimeoutError("MCP call timed out"))
        else:
            request.result.set_result({"diagnostics": []})

    monkeypatch.setattr(contract, "_submit_without_blocking_loop", submit)

    async def run():
        return await pool_cls()._execute(
            operation="call_tool",
            stage="generation",
            env={},
            timeout_seconds=1.0,
            name="java_diagnostics",
            arguments={},
        )

    assert anyio.run(run) == {"diagnostics": []}
    assert attempts == 2
    output = capsys.readouterr().err
    assert '"event":"mcp_verifier_transport_retry"' in output
    assert '"event":"mcp_verifier_transport_recovered"' in output


def test_deterministic_verifier_error_is_not_retried(monkeypatch) -> None:
    pool_cls = _installed_fake_pool()
    attempts = 0

    async def submit(_worker, request):
        nonlocal attempts
        attempts += 1
        request.result.set_exception(RuntimeError("invalid schema for verifier argument"))

    monkeypatch.setattr(contract, "_submit_without_blocking_loop", submit)

    async def run():
        return await pool_cls()._execute(
            operation="call_tool",
            stage="generation",
            env={},
            timeout_seconds=1.0,
            name="java_diagnostics",
            arguments={},
        )

    with pytest.raises(RuntimeError, match="invalid schema"):
        anyio.run(run)
    assert attempts == 1


def test_non_verifier_timeout_is_never_retried(monkeypatch) -> None:
    pool_cls = _installed_fake_pool()
    attempts = 0

    async def submit(_worker, request):
        nonlocal attempts
        attempts += 1
        request.result.set_exception(TimeoutError("mutation timed out"))

    monkeypatch.setattr(contract, "_submit_without_blocking_loop", submit)

    async def run():
        return await pool_cls()._execute(
            operation="call_tool",
            stage="generation",
            env={},
            timeout_seconds=1.0,
            name="apply_source_patch",
            arguments={},
        )

    with pytest.raises(TimeoutError, match="mutation timed out"):
        anyio.run(run)
    assert attempts == 1


def test_three_consecutive_transient_failures_fail_closed_with_history(monkeypatch) -> None:
    pool_cls = _installed_fake_pool()
    attempts = 0

    async def submit(_worker, request):
        nonlocal attempts
        attempts += 1
        request.result.set_exception(ConnectionResetError("MCP channel closed"))

    monkeypatch.setattr(contract, "_submit_without_blocking_loop", submit)

    async def run():
        return await pool_cls()._execute(
            operation="call_tool",
            stage="generation",
            env={},
            timeout_seconds=1.0,
            name="java_diagnostics",
            arguments={},
        )

    with pytest.raises(AgentToolRuntimeError, match="MCP_TRANSIENT_RETRY_EXHAUSTED") as raised:
        anyio.run(run)
    assert attempts == contract._MCP_MAX_CONSECUTIVE_INFRA_FAILURES
    assert "retry_history=" in str(raised.value)
    assert isinstance(raised.value.__cause__, ConnectionResetError)


def test_timeout_remains_primary_when_cleanup_error_is_secondary() -> None:
    timeout = TimeoutError("MCP call timed out")
    timeout.__context__ = RuntimeError("no running event loop")
    assert contract._is_transient_mcp_transport_failure(timeout) is True
