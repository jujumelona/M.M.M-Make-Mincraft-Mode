from __future__ import annotations

"""Canonical classification for semantic-task execution ownership.

Planning, execution lowering, and preflight linking must agree on what constitutes a
runtime capability and on whether an anchor is production or test-only. Keeping these
rules here prevents a task from being accepted by one stage and rejected by the next.
"""

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

SOURCE_ROOTS = (
    "src/main/java/",
    "src/client/java/",
)
TEST_ROOTS = (
    "src/test/",
    "src/gametest/",
)
_NON_RUNTIME_OUTCOMES = frozenset({"test", "verification", "resource", "asset"})


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def anchor_locator(anchor: Mapping[str, Any]) -> str:
    return str(anchor.get("locator") or "").replace("\\", "/").strip()


def path_from_locator(locator: str) -> str:
    raw = str(locator or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or ":" in raw.split("#", 1)[0]:
        return ""
    path = raw.split("#", 1)[0]
    while path.startswith("./"):
        path = path[2:]
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    return PurePosixPath(path).as_posix()


def is_source_symbol(anchor: Mapping[str, Any]) -> bool:
    if str(anchor.get("kind") or "").strip().casefold() != "symbol":
        return False
    path = path_from_locator(anchor_locator(anchor))
    return bool(path and path.endswith(".java") and path.startswith(SOURCE_ROOTS))


def is_test_anchor(anchor: Mapping[str, Any]) -> bool:
    if str(anchor.get("kind") or "").strip().casefold() == "test":
        return True
    path = path_from_locator(anchor_locator(anchor))
    return bool(path and path.startswith(TEST_ROOTS))


def claims_runtime(task: Mapping[str, Any]) -> bool:
    provides = _strings(task.get("provides"))
    if any(value.startswith("capability:") for value in provides):
        return True
    semantic = str(task.get("semantic_outcome") or "").strip().casefold()
    return bool(semantic and semantic not in _NON_RUNTIME_OUTCOMES)


__all__ = [
    "SOURCE_ROOTS",
    "TEST_ROOTS",
    "anchor_locator",
    "claims_runtime",
    "is_source_symbol",
    "is_test_anchor",
    "path_from_locator",
]
