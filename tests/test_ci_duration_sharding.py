from __future__ import annotations

from tools.ci_test_shard import partition_by_duration, select_shard


def test_duration_partition_is_complete_disjoint_and_deterministic() -> None:
    paths = [f"tests/test_{index}.py" for index in range(9)]
    durations = {path: float(index + 1) for index, path in enumerate(paths)}
    first = partition_by_duration(paths, shard_count=3, durations=durations)
    second = partition_by_duration(reversed(paths), shard_count=3, durations=durations)
    assert first == second
    flattened = [path for shard in first for path in shard]
    assert sorted(flattened) == sorted(paths)
    assert len(flattened) == len(set(flattened))


def test_duration_partition_balances_skewed_runtime() -> None:
    paths = ["tests/slow.py", "tests/a.py", "tests/b.py", "tests/c.py"]
    durations = {"tests/slow.py": 9.0, "tests/a.py": 3.0, "tests/b.py": 3.0, "tests/c.py": 3.0}
    shards = partition_by_duration(paths, shard_count=2, durations=durations)
    loads = [sum(durations[path] for path in shard) for shard in shards]
    assert max(loads) - min(loads) <= 3.0


def test_select_shard_uses_duration_partition_when_history_supplied() -> None:
    paths = ["tests/a.py", "tests/b.py", "tests/c.py"]
    durations = {"tests/a.py": 10.0, "tests/b.py": 1.0, "tests/c.py": 1.0}
    expected = partition_by_duration(paths, shard_count=2, durations=durations)
    assert select_shard(paths, shard_number=1, shard_count=2, durations=durations) == expected[0]
    assert select_shard(paths, shard_number=2, shard_count=2, durations=durations) == expected[1]
