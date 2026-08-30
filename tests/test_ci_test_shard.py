from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci_test_shard import select_shard, shard_index


def test_assignment_is_stable_when_unrelated_tests_are_added_or_removed() -> None:
    baseline = [
        "tests/test_alpha.py",
        "tests/test_beta.py",
        "tests/test_gamma.py",
        "tests/test_delta.py",
    ]
    assignments = {path: shard_index(path, shard_count=3) for path in baseline}

    changed_inventory = [
        "tests/test_aaa_new.py",
        *reversed(baseline),
        "tests/test_zzz_new.py",
    ]
    changed_assignments = {
        path: shard_index(path, shard_count=3)
        for path in changed_inventory
        if path in assignments
    }
    assert changed_assignments == assignments


def test_select_shard_is_order_independent_and_deduplicates_paths() -> None:
    inventory = [
        "tests/test_c.py",
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_a.py",
    ]
    selected = select_shard(inventory, shard_number=2, shard_count=3)
    assert selected == tuple(sorted(set(selected)))
    assert all(shard_index(path, shard_count=3) == 1 for path in selected)
    assert select_shard(reversed(inventory), shard_number=2, shard_count=3) == selected


def test_three_shards_are_disjoint_and_cover_inventory() -> None:
    inventory = [f"tests/test_{index}.py" for index in range(50)]
    shards = [
        set(select_shard(inventory, shard_number=number, shard_count=3))
        for number in (1, 2, 3)
    ]
    assert shards[0].isdisjoint(shards[1])
    assert shards[0].isdisjoint(shards[2])
    assert shards[1].isdisjoint(shards[2])
    assert set().union(*shards) == set(inventory)


def test_invalid_shard_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="shard_count"):
        shard_index("tests/test_a.py", shard_count=0)
    with pytest.raises(ValueError, match="shard_number"):
        select_shard(["tests/test_a.py"], shard_number=0, shard_count=3)
    with pytest.raises(ValueError, match="shard_number"):
        select_shard(["tests/test_a.py"], shard_number=4, shard_count=3)


def test_ci_workflow_uses_stable_shard_helper() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "from tools.ci_test_shard import select_shard" in workflow
    assert "select_shard(" in workflow
    assert "enumerate(all_tests)" not in workflow
