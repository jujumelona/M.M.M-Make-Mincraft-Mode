"""Bounded warm external MCP sessions owned by one construction operation.

Every provider session stays inside its portal event loop; never move an MCP
ClientSession across event loops or threads. Both successful and failed tool
calls retain the reviewed provider identity and runtime target checks in the
parent ExternalMCPRouter.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager, AsyncExitStack
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
                args=[str(v) for v in command[1:]], env=dict(env))
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
        self.connection = None
        self.session = None
        self.info = None
        self.listed = None

    async def _call(self, tool, arguments, timeout):
        from .external_mcp_router import (
            ExternalMCPError, _checked_provider_call, _jsonable, _normalize_tool_result,
        )
        with anyio.fail_after(timeout):
            if self.session is None:
                self.connection = self.connector(
                    self.name, self.entry, self.env, self.url)
                self.session, self.info, self.listed = await self.connection.__aenter__()
            available = {str(item.name) for item in getattr(self.listed, "tools", ())}
            if tool not in available:
                raise ExternalMCPError(
                    f"Provider does not expose reviewed tool {tool!r}; available={sorted(available)}")
            raw = await _checked_provider_call(
                self.session, tool, arguments, listed=self.listed,
                prepare_minecraft_source=(self.name == "minecraft-dev"))
            return {"server_info": _jsonable(getattr(self.info, "serverInfo", None)),
                    "result": _normalize_tool_result(raw)}

    async def _disconnect(self):
        if self.connection is not None:
            connection = self.connection
            self.connection = None
            self.session = None
            self.listed = None
            self.info = None
            await connection.__aexit__(None, None, None)

    def call(self, tool, arguments, timeout):
        with self.lock:
            if self.portal is None:
                self.portal_context = start_blocking_portal(
                    name="mmm-ext-mcp-" + str(self.name))
                self.portal = self.portal_context.__enter__()
            try:
                return self.portal.call(
                    self._call, tool, dict(arguments), float(timeout))
            except BaseException:
                # A broken connection must never be reused. Keep subsequent
                # route fallback available to the existing router.
                try:
                    self.portal.call(self._disconnect)
                except BaseException:
                    pass
                raise

    def close(self):
        with self.lock:
            if self.portal is None:
                return
            try:
                self.portal.call(self._disconnect)
            finally:
                self.portal_context.__exit__(None, None, None)
                self.portal = None
                self.portal_context = None


class WarmProviderPool:
    """At most two reusable providers for an active coding file.

    The parent construction context closes this pool at every success/error
    boundary. This is not a global background daemon or unbounded process pool.
    """

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
                          url, sorted(env.items())], default=str)
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
