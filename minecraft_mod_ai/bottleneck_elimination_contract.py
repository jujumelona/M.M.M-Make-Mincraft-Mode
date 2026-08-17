from __future__ import annotations

"""Retired compatibility entry point.

Runtime ownership now lives in the modules that implement llama generation, research,
and MCP transport.  This module intentionally installs no cross-module monkeypatches;
it remains only until the bootstrap import is removed so live environments upgrading
in place do not fail on an import during the transition.
"""


def install() -> None:
    """No-op: monolithic runtime monkeypatch composition has been retired."""


__all__ = ["install"]
