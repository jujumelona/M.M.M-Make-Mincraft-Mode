from __future__ import annotations

"""Retired late-bootstrap research monkeypatch composition.

Research, retrieval, memory, validation and MCP behavior are owned by their source
modules. Runtime bootstrap may still import this transition entry point while older
live environments upgrade, but it intentionally mutates nothing.
"""


def install() -> None:
    """No-op: cross-module research hotpath patching has been retired."""


__all__ = ["install"]
