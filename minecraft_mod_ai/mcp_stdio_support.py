from __future__ import annotations

"""Shared stdio subprocess support for first-party and external MCP clients."""

import tempfile
from typing import IO


def open_mcp_stdio_errlog() -> IO[str]:
    """Return a real fd-backed text stream suitable for ``mcp.stdio_client``.

    Colab/IPython replaces ``sys.stderr`` with objects that may not implement a usable
    ``fileno()``. MCP eventually passes the error stream to subprocess creation, so
    every stdio transport must use a real OS-backed descriptor. The caller owns the
    returned context-managed file object.
    """

    return tempfile.TemporaryFile(mode="w+", encoding="utf-8")


__all__ = ["open_mcp_stdio_errlog"]
