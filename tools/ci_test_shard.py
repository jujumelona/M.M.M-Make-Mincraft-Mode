from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path


def _normalize(paths: Iterable[str | Path]) -> tuple[str, ...]:
    return tuple(sorted({Path(path).as_posix() for path in paths}))


def shard_index(path: str | Path, *, shard_count: int) -> int:
    """Return a stable zero-based shard index for a repository-relative test path."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    normalized = Path(path).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def partition_by_duration(
    paths: Iterable[str | Path],
    *,
    shard_count: int,
    durations: Mapping[str, float] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Partition tests deterministically using longest-processing-time scheduling.

    Known historical durations drive balancing. Unknown files receive the median known
    duration (or 1.0 when no history exists), so new tests do not collapse onto one
    shard. Path and shard index provide deterministic tie-breaking.
    """

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    normalized = _normalize(paths)
    history = {
        Path(path).as_posix(): float(seconds)
        for path, seconds in (durations or {}).items()
        if float(seconds) >= 0.0
    }
    known = sorted(history[path] for path in normalized if path in history)
    if known:
        middle = len(known) // 2
        fallback = known[middle] if len(known) % 2 else (known[middle - 1] + known[middle]) / 2.0
    else:
        fallback = 1.0

    weighted = [(path, history.get(path, fallback)) for path in normalized]
    weighted.sort(key=lambda item: (-item[1], item[0]))
    loads = [0.0] * shard_count
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for path, seconds in weighted:
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[target].append(path)
        loads[target] += seconds
    return tuple(tuple(sorted(shard)) for shard in shards)


def select_shard(
    paths: Iterable[str | Path],
    *,
    shard_number: int,
    shard_count: int,
    durations: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Select a stable one-based shard, duration-balanced when history is supplied."""

    if not 1 <= shard_number <= shard_count:
        raise ValueError(
            f"shard_number must be between 1 and {shard_count}, got {shard_number}"
        )
    if durations is not None:
        return partition_by_duration(
            paths,
            shard_count=shard_count,
            durations=durations,
        )[shard_number - 1]
    target = shard_number - 1
    return tuple(
        path for path in _normalize(paths) if shard_index(path, shard_count=shard_count) == target
    )
