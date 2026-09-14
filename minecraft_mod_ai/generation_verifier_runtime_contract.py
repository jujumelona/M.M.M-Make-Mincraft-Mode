from __future__ import annotations

"""Lifecycle contract for the persistent host-owned generation JDT verifier.

Incremental targeting belongs to the persistent JDT Core workspace builder.
This module owns only runtime shutdown so the
persistent JDT process cannot leak beyond the runtime that created it.
"""

from functools import wraps
from typing import Any

_CLOSE_MARKER = "_mmm_generation_verifier_close"


def install(*, agent_tool_runtime_module: Any, verifier_module: Any) -> None:
    """Close the persistent generation JDT service with its owning runtime."""

    # Keep the explicit dependency in the installation contract: runtime finalization
    # passes the verifier owner here, but targeting/fallback mutation is intentionally gone.
    if verifier_module is None:
        raise TypeError("verifier_module is required")

    runtime_cls = agent_tool_runtime_module.AgentToolRuntime
    current_close = runtime_cls.close
    if getattr(current_close, _CLOSE_MARKER, False):
        return

    @wraps(current_close)
    def close_with_generation_verifier(self: Any) -> None:
        service = getattr(self, "_mmm_generation_java_service", None)
        if service is not None:
            try:
                close = getattr(service, "close", None)
                if callable(close):
                    close()
            finally:
                try:
                    delattr(self, "_mmm_generation_java_service")
                except AttributeError:
                    pass
        current_close(self)

    setattr(close_with_generation_verifier, _CLOSE_MARKER, True)
    close_with_generation_verifier.__wrapped__ = current_close
    runtime_cls.close = close_with_generation_verifier


__all__ = ["install"]
