from __future__ import annotations

"""Retired compatibility installer.

Causal action selection is now owned by the frontier and writable actions are
materialized by the host-selected argument protocol in
``forced_tool_execution_contract``.  Re-asking the model to select a stale/forced tool
name is intentionally unsupported.

This temporary no-op remains only until bootstrap references are removed in the same
cleanup series; it must not wrap or retry any model call.
"""


def install(_causal_frontier_adapter_module: object) -> None:
    return None


__all__ = ["install"]
