from __future__ import annotations

from tools.ci_test_matrix import build_matrix


def test_matrix_uses_one_job_per_file_below_cap() -> None:
    paths = [f"tests/test_{index:03d}.py" for index in range(20)]
    matrix = build_matrix(paths, max_jobs=256)
    assert len(matrix) == len(paths)
    assert sorted(file for row in matrix for file in row["files"]) == sorted(paths)
    assert all(len(row["files"]) == 1 for row in matrix)


def test_matrix_caps_jobs_and_preserves_exact_coverage() -> None:
    paths = [f"tests/test_{index:03d}.py" for index in range(300)]
    matrix = build_matrix(paths, max_jobs=256)
    flattened = [file for row in matrix for file in row["files"]]
    assert len(matrix) == 256
    assert len(flattened) == len(set(flattened)) == 300
    assert sorted(flattened) == sorted(paths)


def test_matrix_is_deterministic_and_duration_balanced() -> None:
    paths = [
        "tests/test_slow.py",
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_c.py",
    ]
    durations = {
        "tests/test_slow.py": 12.0,
        "tests/test_a.py": 1.0,
        "tests/test_b.py": 1.0,
        "tests/test_c.py": 1.0,
    }
    first = build_matrix(paths, max_jobs=2, durations=durations)
    second = build_matrix(list(reversed(paths)), max_jobs=2, durations=durations)
    assert first == second
    loads = [sum(durations[file] for file in row["files"]) for row in first]
    assert sorted(loads) == [3.0, 12.0]
