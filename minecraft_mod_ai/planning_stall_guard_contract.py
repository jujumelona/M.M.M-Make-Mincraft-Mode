from __future__ import annotations

"""Retired planner stall monkeypatch compatibility hook."""


def install() -> None:
    """No-op: planning progress now belongs to host-owned planner stages."""
    return


__all__ = ["install"]
