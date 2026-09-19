from __future__ import annotations

"""Prevent stale validation receipts from triggering an unrelated execution replay."""

import sys


def _checkpoint_for_exception(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    message = str(exc).casefold()
    if "failed deterministic validation" in message:
        return "validate-source"
    if "jdt reported errors" in message:
        return "validate-jdt"
    if "gradle/gametest failed after the repair loop" in message:
        return "gradle-build"
    # Artifact/runtime/visual/publication failures must never resurrect an older
    # source/JDT/build failure merely because its receipt remains in the ledger.
    return None


def _active_exception() -> BaseException | None:
    # sys.exception() is Python 3.11+. MMM still supports Python 3.10, where the
    # exception currently handled by the calling thread is available via exc_info().
    getter = getattr(sys, "exception", None)
    if callable(getter):
        return getter()
    return sys.exc_info()[1]



__all__ = ["_active_exception", "_checkpoint_for_exception"]
