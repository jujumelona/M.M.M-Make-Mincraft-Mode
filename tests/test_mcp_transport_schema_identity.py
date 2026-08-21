from __future__ import annotations

import asyncio
from concurrent.futures import Future
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.mcp_transport_pool import (
    MCPTransportPool,
    _SessionWorker,
    _TransportRequest,
)


def _listed(property_name: str):
    return SimpleNamespace(
        tools=(
            SimpleNamespace(
                name="lookup",
                description="lookup",
                inputSchema={
                    "type": "object",
                    "properties": {property_name: {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
        )
    )


def test_pooled_worker_schema_drift_blocks_execution_before_call_tool() -> None:
    state = {"property": "query", "calls": 0}

    class Session:
        def __init__(self, property_name: str) -> None:
            self.property_name = property_name

        async def list_tools(self):
            return _listed(self.property_name)

        async def call_tool(self, name, *, arguments):
            state["calls"] += 1
            return {"name": name, "arguments": dict(arguments)}

    @asynccontextmanager
    async def session_factory(stage, env, timeout_seconds):
        yield Session(state["property"])

    async def scenario() -> None:
        pool = MCPTransportPool(worker_count=2, session_factory=session_factory)
        try:
            first = await pool.list_tools(
                stage="generation",
                env={"MMM_MCP_STAGE": "generation"},
                timeout_seconds=5.0,
            )
            assert first.tools[0].inputSchema["properties"] == {
                "query": {"type": "string"}
            }

            # Force the next dispatch to the second worker. It starts after the
            # provider schema has changed, so execution must fail before call_tool.
            pool._workers[0].reserve()
            state["property"] = "symbol"
            with pytest.raises(RuntimeError, match="pooled worker schema drift"):
                await pool.call_tool(
                    stage="generation",
                    env={"MMM_MCP_STAGE": "generation"},
                    timeout_seconds=5.0,
                    name="lookup",
                    arguments={"query": "BlockEntity"},
                )
            assert state["calls"] == 0
        finally:
            pool.close()

    asyncio.run(scenario())


def test_pooled_schema_identity_allows_same_schema_across_workers() -> None:
    state = {"calls": 0}

    class Session:
        async def list_tools(self):
            return _listed("query")

        async def call_tool(self, name, *, arguments):
            state["calls"] += 1
            return SimpleNamespace(isError=False, structuredContent={"ok": True})

    @asynccontextmanager
    async def session_factory(stage, env, timeout_seconds):
        yield Session()

    async def scenario() -> None:
        pool = MCPTransportPool(worker_count=2, session_factory=session_factory)
        try:
            await pool.list_tools(
                stage="generation",
                env={"MMM_MCP_STAGE": "generation"},
                timeout_seconds=5.0,
            )
            pool._workers[0].reserve()
            await pool.call_tool(
                stage="generation",
                env={"MMM_MCP_STAGE": "generation"},
                timeout_seconds=5.0,
                name="lookup",
                arguments={"query": "BlockEntity"},
            )
            assert state["calls"] == 1
        finally:
            pool.close()

    asyncio.run(scenario())


def test_environment_change_uses_independent_schema_identity_scope() -> None:
    async def scenario() -> None:
        pool = MCPTransportPool(worker_count=1)
        try:
            pool._register_schema(
                stage="generation",
                env={"MMM_MCP_STAGE": "generation", "FEATURE": "a"},
                timeout_seconds=5.0,
                listed=_listed("query"),
            )
            # A changed child environment is a different MCP process contract and
            # must not be compared to the old environment's canonical fingerprint.
            pool._register_schema(
                stage="generation",
                env={"MMM_MCP_STAGE": "generation", "FEATURE": "b"},
                timeout_seconds=5.0,
                listed=_listed("symbol"),
            )
            assert len(pool._schema_fingerprints) == 2
        finally:
            pool.close()

    asyncio.run(scenario())


def test_cancelled_queued_request_never_opens_session_or_executes_tool() -> None:
    state = {"sessions": 0, "calls": 0}

    class Session:
        async def list_tools(self):
            return _listed("query")

        async def call_tool(self, name, *, arguments):
            state["calls"] += 1
            return {"ok": True}

    @asynccontextmanager
    async def session_factory(stage, env, timeout_seconds):
        state["sessions"] += 1
        yield Session()

    worker = _SessionWorker(
        session_factory=session_factory,
        max_pending=2,
        name="mmm-test-cancelled-request",
    )
    future: Future[object] = Future()
    worker.reserve()
    future.cancel()
    request = _TransportRequest(
        operation="call_tool",
        stage="generation",
        env={"MMM_MCP_STAGE": "generation"},
        timeout_seconds=2.0,
        result=future,
        name="lookup",
        arguments={"query": "BlockEntity"},
        expected_schema_sha256="unused-because-cancelled",
    )

    try:
        worker.submit(request)
    finally:
        worker.close()

    assert worker.pending == 0
    assert state == {"sessions": 0, "calls": 0}
