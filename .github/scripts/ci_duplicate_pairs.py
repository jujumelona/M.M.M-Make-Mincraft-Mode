from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable


def duplicate_pairs(groups: Iterable[Iterable[Any]]) -> set[tuple[Any, Any]]:
    """Expand duplicate groups into pairwise relations.

    Group membership may shrink after code deletion without creating a new duplicate.
    A regression exists only when a pair of members was not duplicated in the baseline.
    """

    pairs: set[tuple[Any, Any]] = set()
    for group in groups:
        members = sorted(set(group))
        pairs.update(combinations(members, 2))
    return pairs


def introduced_duplicate_pairs(
    current_groups: Iterable[Iterable[Any]],
    baseline_groups: Iterable[Iterable[Any]],
) -> list[tuple[Any, Any]]:
    return sorted(duplicate_pairs(current_groups) - duplicate_pairs(baseline_groups))


__all__ = ["duplicate_pairs", "introduced_duplicate_pairs"]
