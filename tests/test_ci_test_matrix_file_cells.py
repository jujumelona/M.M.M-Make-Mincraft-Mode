from __future__ import annotations

import pytest

from tools.ci_test_matrix import build_file_lanes


def test_file_lanes_put_exactly_one_unique_test_in_each_cell() -> None:
    paths = [f"tests/test_{index:03}.py" for index in range(752)]

    lanes = build_file_lanes(paths, lane_count=4, max_jobs_per_lane=256)

    assert len(lanes) == 4
    cells = [cell for lane in lanes for cell in lane]
    assert len(cells) == 752
    assert all(len(cell["files"]) == 1 for cell in cells)
    assert sorted(cell["files"][0] for cell in cells) == paths
    assert max(len(lane) for lane in lanes) <= 256


def test_file_lanes_are_deterministic_and_deduplicate_paths() -> None:
    paths = ["tests/test_b.py", "tests/test_a.py", "tests/test_b.py"]

    first = build_file_lanes(paths, lane_count=4, max_jobs_per_lane=256)
    second = build_file_lanes(list(reversed(paths)), lane_count=4, max_jobs_per_lane=256)

    assert first == second
    assert sorted(cell["files"][0] for lane in first for cell in lane) == [
        "tests/test_a.py",
        "tests/test_b.py",
    ]


def test_file_lanes_fail_closed_when_capacity_is_exceeded() -> None:
    paths = [f"tests/test_{index:04}.py" for index in range(1025)]

    with pytest.raises(ValueError, match="exceed file-cell capacity"):
        build_file_lanes(paths, lane_count=4, max_jobs_per_lane=256)


def test_file_lanes_use_full_matrix_capacity_without_grouping() -> None:
    paths = [f"tests/test_{index:04}.py" for index in range(1024)]

    lanes = build_file_lanes(paths, lane_count=4, max_jobs_per_lane=256)

    assert [len(lane) for lane in lanes] == [256, 256, 256, 256]
    assert all(len(cell["files"]) == 1 for lane in lanes for cell in lane)
