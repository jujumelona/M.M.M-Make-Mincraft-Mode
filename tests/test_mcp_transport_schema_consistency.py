from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any, Mapping

import pytest

from minecraft_mod_ai.mcp_transport_pool import MCPTransportPool


class _Tool:
    def __init__(self, name: str, marker: str) -> None:
        self.name = name
        self.description = marker
        self.inputSchema = {
            "type": "object",
            "properties": {
                "marker": {"type": "string", "description": marker},
            },
            "additionalProperties": False,
        }


class _Listed:
    def __init__(self, marker: str) -> None:
        self.tools = (_Tool("lookup", marker),)


class _SchemaSession:
    def __init__(
        self,
        marker: str,
        *,
        block_event: asyncio.Event | None = None,
        fail_first_call: bool = False,
    ) -> None:
        self.marker = marker
        self.block_event = block_event
        self.fail_first_call = fail_first_call
        self.call_count = 0

    async def list_tools(self) -> _Listed:
        return _Listed(self.marker)

    async def call_tool(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.call_count += 1
        if self.fail_first_call and self.call_count == 1:
            raise RuntimeError("synthetic transport failure")
        if self.block_event is not None:
            await self.block_event.wait()
        return {"name": name, "arguments": dict(arguments), "marker": self.marker}


class _SessionContext(AbstractAsyncContextManager[_SchemaSession]):
    def __init__(self, session: _SchemaSession) -> None:
        self.session = session

    async def __aenter__(self) -> _SchemaSession:
        return self.session

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def test_second_pooled_worker_with_different_schema_is_blocked_before_tool_execution() -> None:
    async def exercise() -> None:
        release = asyncio.Event()
        opened = 0
        sessions: list[_SchemaSession] = []

        def factory(stage: str, env: Mapping[str, str], timeout_seconds: float):
            nonlocal opened
            del stage, env, timeout_seconds
            opened += 1
            session = _SchemaSession(
                "schema-a" if opened == 1 else "schema-b",
                block_event=release if opened == 1 else None,
            )
            sessions.append(session)
            return _SessionContext(session)

        pool = MCPTransportPool(worker_count=2, session_factory=factory)
        try:
            await pool.list_tools(
                stage="research",
                env={"FEATURE": "stable"},
                timeout_seconds=1.0,
            )
            first = asyncio.create_task(
                pool.call_tool(
                    stage="research",
                    env={"FEATURE": "stable"},
                    timeout_seconds=1.0,
                    name="lookup",
                    arguments={"request": 1},
                )
            )
            # Let worker 0 reserve the live call so the second dispatch must use worker 1.
            await asyncio.sleep(0.02)
            with pytest.raises(RuntimeError, match="schema drift detected before execution"):
                await pool.call_tool(
                    stage="research",
                    env={"FEATURE": "stable"},
                    timeout_seconds=1.0,
                    name="lookup",
                    arguments={"request": 2},
                )
            assert len(sessions) == 2
            assert sessions[1].call_count == 0
            release.set()
            assert (await first)["marker"] == "schema-a"
        finally:
            release.set()
            pool.close()

    asyncio.run(exercise())


def test_worker_restart_with_changed_schema_cannot_reuse_old_canonical_contract() -> None:
    async def exercise() -> None:
        opened = 0
        sessions: list[_SchemaSession] = []

        def factory(stage: str, env: Mapping[str, str], timeout_seconds: float):
            nonlocal opened
            del stage, env, timeout_seconds
            opened += 1
            session = _SchemaSession(
                "schema-a" if opened == 1 else "schema-b",
                fail_first_call=opened == 1,
            )
            sessions.append(session)
            return _SessionContext(session)

        pool = MCPTransportPool(worker_count=1, session_factory=factory)
        try:
            await pool.list_tools(
                stage="generation",
                env={"FEATURE": "stable"},
                timeout_seconds=1.0,
            )
            with pytest.raises(RuntimeError, match="synthetic transport failure"):
                await pool.call_tool(
                    stage="generation",
                    env={"FEATURE": "stable"},
                    timeout_seconds=1.0,
                    name="lookup",
                    arguments={},
                )
            with pytest.raises(RuntimeError, match="schema drift detected before execution"):
                await pool.call_tool(
                    stage="generation",
                    env={"FEATURE": "stable"},
                    timeout_seconds=1.0,
                    name="lookup",
                    arguments={},
                )
            assert len(sessions) == 2
            assert sessions[1].call_count == 0
        finally:
            pool.close()

    asyncio.run(exercise())
