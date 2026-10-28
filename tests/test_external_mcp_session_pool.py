"""Warm MCP session contract: same-task lifetime, reuse, fallback and eviction."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.external_mcp_session_pool import WarmProviderPool


def _fake_connector(events):
    @asynccontextmanager
    async def connect(server, _entry, _env, _url):
        events.append(("enter", server, id(asyncio.current_task())))

        class Session:
            async def call_tool(self, tool, *, arguments):
                events.append(("call", server, id(asyncio.current_task())))
                if arguments.get("fail"):
                    raise ValueError("provider execution failed")
                return SimpleNamespace(content=[SimpleNamespace(text="verified lookup page")],
                                       structuredContent=None, isError=False)

        try:
            yield Session(), SimpleNamespace(serverInfo={"name": server}), SimpleNamespace(
                tools=[SimpleNamespace(name="search", inputSchema={"type": "object"})])
        finally:
            events.append(("exit", server, id(asyncio.current_task())))

    return connect


def _run(pool, server="docs", **args):
    return pool.call(server, {"transport": "stdio", "command": ["test-provider"]},
        tool="search", arguments=args, timeout=10, env={"PATH": "/usr/bin"}, url="")


def test_warm_calls_reuse_one_connection_and_close_on_owning_task():
    events = []
    pool = WarmProviderPool(connector=_fake_connector(events))
    try:
        assert _run(pool)["result"]["text"] == ["verified lookup page"]
        assert _run(pool)["result"]["text"] == ["verified lookup page"]
    finally:
        pool.close()
    assert [e[0] for e in events] == ["enter", "call", "call", "exit"]
    assert len({event[2] for event in events}) == 1
    with pytest.raises(RuntimeError, match="closed"):
        _run(pool)


def test_provider_error_discards_session_then_reconnects():
    events = []
    pool = WarmProviderPool(connector=_fake_connector(events))
    try:
        with pytest.raises(ValueError, match="provider execution failed"):
            _run(pool, fail=True)
        assert _run(pool)["result"]["text"] == ["verified lookup page"]
    finally:
        pool.close()
    assert [e[0] for e in events] == [
        "enter", "call", "exit", "enter", "call", "exit",
    ]


def test_max_sessions_evicts_old_provider_and_closes_it():
    events = []
    pool = WarmProviderPool(max_providers=1, connector=_fake_connector(events))
    try:
        _run(pool, "one")
        _run(pool, "two")
        _run(pool, "two")
    finally:
        pool.close()
    assert [e[0:2] for e in events] == [
        ("enter", "one"), ("call", "one"), ("exit", "one"),
        ("enter", "two"), ("call", "two"), ("call", "two"), ("exit", "two"),
    ]
