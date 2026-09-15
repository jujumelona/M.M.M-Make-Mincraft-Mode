from __future__ import annotations

"""Shared stdio subprocess support for first-party and external MCP clients."""

import builtins
import sys
import tempfile
from functools import wraps
from typing import IO, Any

_PRINT_GUARD_MARKER = "_mmm_mcp_protocol_print_guard_v1"


def install_mcp_protocol_print_guard() -> None:
    """Reserve process stdout for MCP JSON-RPC frames.

    MCP stdio transports own stdout.  Runtime diagnostics in MMM still use ordinary
    ``print`` in several deep execution paths, so a server process must redirect the
    default print destination before importing/running those paths.  Explicit ``file=``
    destinations remain untouched and the MCP transport's direct stdout stream writes
    are unaffected.
    """

    current = builtins.print
    if getattr(current, _PRINT_GUARD_MARKER, False):
        return

    @wraps(current)
    def protocol_safe_print(*args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("file", sys.stderr)
        current(*args, **kwargs)

    setattr(protocol_safe_print, _PRINT_GUARD_MARKER, True)
    builtins.print = protocol_safe_print


def open_mcp_stdio_errlog() -> IO[str]:
    """Return a real fd-backed text stream suitable for ``mcp.stdio_client``.

    Colab/IPython replaces ``sys.stderr`` with objects that may not implement a usable
    ``fileno()``. MCP eventually passes the error stream to subprocess creation, so
    every stdio transport must use a real OS-backed descriptor. The caller owns the
    returned context-managed file object.
    """

    return tempfile.TemporaryFile(mode="w+", encoding="utf-8")


__all__ = ["install_mcp_protocol_print_guard", "open_mcp_stdio_errlog"]
