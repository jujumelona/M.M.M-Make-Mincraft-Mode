from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def shard_index(path: str | Path, *, shard_count: int) -> int:
    """Return a stable zero-based shard index for a repository-relative test path."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    normalized = Path(path).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def select_shard(
    paths: Iterable[str | Path],
    *,
    shard_number: int,
    shard_count: int,
) -> tuple[str, ...]:
    """Select a stable one-based shard without order-dependent reassignment."""

    if not 1 <= shard_number <= shard_count:
        raise ValueError(
            f"shard_number must be between 1 and {shard_count}, got {shard_number}"
        )
    target = shard_number - 1
    normalized = sorted({Path(path).as_posix() for path in paths})
    return tuple(
        path for path in normalized if shard_index(path, shard_count=shard_count) == target
    )
