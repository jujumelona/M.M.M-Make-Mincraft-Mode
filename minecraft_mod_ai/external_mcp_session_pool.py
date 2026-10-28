"""Warm external MCP sessions with bounded provider count and explicit lifetime.

MCP session context managers must open, execute, and exit in the SAME asyncio
task. The portal only submits commands to a long-running provider actor.
"""
from __future__ import annotations

import asyncio
import json
import threading
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import anyio
from anyio.from_thread import start_blocking_portal

from .mcp_stdio_support import open_mcp_stdio_errlog


@asynccontextmanager
async def _connect_provider(server_name, entry, env, url):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    async with AsyncExitStack() as stack:
        if entry.get("transport") == "stdio":
            command = entry.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError("Invalid external stdio MCP command")
            errlog = stack.enter_context(open_mcp_stdio_errlog())
            params = StdioServerParameters(command=str(command[0]),
                args=[str(value) for value in command[1:]], env=dict(env))
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
        elif entry.get("transport") == "streamable_http":
            if not url:
                raise ValueError("External HTTP MCP URL is missing")
            read, write, _ = await stack.enter_async_context(streamable_http_client(url))
        else:
            raise ValueError("Unsupported external MCP transport")
        session = await stack.enter_async_context(ClientSession(read, write))
        info = await session.initialize()
        listed = await session.list_tools()
        yield session, info, listed


class _WarmProvider:
    def __init__(self, name, entry, env, url, connector):
        self.name, self.entry, self.env, self.url = name, entry, env, url
        self.connector = connector
        self.lock = threading.RLock()
        self.portal_context = None
        self.portal = None
        self.queue = None
        self.actor = None

    def _start_actor(self):
        self.queue = asyncio.Queue()
        self.actor = asyncio.create_task(self._serve())

    def _submit(self, tool, arguments, timeout):
        future = Future()
        self.queue.put_nowait((tool, dict(arguments), float(timeout), future))
        return future

    def _stop(self):
        future = Future()
        self.queue.put_nowait((None, {}, 0.0, future))
        return future

    async def _serve(self):
        from .external_mcp_router import (
            ExternalMCPError, _checked_provider_call, _jsonable, _normalize_tool_result,
        )
        cm = None
        session = info = listed = None
        while True:
            tool, arguments, timeout, future = await self.queue.get()
            if tool is None:
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except BaseException as exc:
                        future.set_exception(exc)
                    else:
                        future.set_result(None)
                else:
                    future.set_result(None)
                return
            try:
                with anyio.fail_after(timeout):
                    if cm is None:
                        new_cm = self.connector(self.name, self.entry, self.env, self.url)
                        session, info, listed = await new_cm.__aenter__()
                        cm = new_cm
                    available = {str(item.name) for item in getattr(listed, "tools", ())}
                    if tool not in available:
                        raise ExternalMCPError(
                            f"Provider does not expose reviewed tool {tool!r}; "
                            f"available={sorted(available)}")
                    raw = await _checked_provider_call(session, tool, arguments,
                        listed=listed, prepare_minecraft_source=(self.name == "minecraft-dev"))
                    value = {
                        "server_info": _jsonable(getattr(info, "serverInfo", None)),
                        "result": _normalize_tool_result(raw),
                    }
                future.set_result(value)
            except BaseException as exc:
                # A provider failure invalidates this connection, not the other
                # providers in the router fallback chain.
                future.set_exception(exc)
                if cm is not None:
                    try:
                        await cm.__aexit__(type(exc), exc, exc.__traceback__)
                    except BaseException:
                        pass
                cm = session = info = listed = None

    def call(self, tool, arguments, timeout):
        with self.lock:
            if self.portal is None:
                self.portal_context = start_blocking_portal(
                    name="mmm-ext-mcp-" + str(self.name))
                self.portal = self.portal_context.__enter__()
                self.portal.call(self._start_actor)
            future = self.portal.call(self._submit, tool, arguments, timeout)
            return future.result(timeout=float(timeout) + 10.0)

    def close(self):
        with self.lock:
            if self.portal is None:
                return
            try:
                self.portal.call(self._stop).result(timeout=20.0)
            finally:
                self.portal_context.__exit__(None, None, None)
                self.portal = None
                self.portal_context = None


class WarmProviderPool:
    """Bounded LRU of live providers, closed at the host construction boundary."""

    def __init__(self, max_providers=2, connector=_connect_provider):
        if type(max_providers) is not int or not 1 <= max_providers <= 4:
            raise ValueError("External MCP pool size must be 1..4")
        self.max_providers = max_providers
        self.connector = connector
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.closed = False

    def call(self, name, entry, *, tool, arguments, timeout, env, url):
        key = json.dumps([name, entry.get("transport"), entry.get("command"),
                          url, sorted(env.items())], sort_keys=True, default=str)
        with self.lock:
            if self.closed:
                raise RuntimeError("External MCP session pool is closed")
            worker = self.entries.get(key)
            if worker is None:
                if len(self.entries) >= self.max_providers:
                    _, stale = self.entries.popitem(last=False)
                    stale.close()
                worker = _WarmProvider(name, entry, env, url, self.connector)
                self.entries[key] = worker
            self.entries.move_to_end(key)
            # Holding the LRU lock through the call prevents another thread
            # from evicting a worker that is still executing a tool.
            return worker.call(tool, arguments, timeout)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            workers = list(self.entries.values())
            self.entries.clear()
            for worker in workers:
                worker.close()
