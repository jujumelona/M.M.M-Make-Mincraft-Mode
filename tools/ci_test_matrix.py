from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.ci_test_shard import partition_by_duration


def build_matrix(
    paths: list[str],
    *,
    max_jobs: int = 256,
    durations: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Build legacy duration-balanced shards for callers that explicitly want grouping."""

    normalized = sorted({Path(path).as_posix() for path in paths})
    if max_jobs < 1:
        raise ValueError("max_jobs must be at least 1")
    if not normalized:
        return []
    groups = partition_by_duration(
        normalized,
        shard_count=min(max_jobs, len(normalized)),
        durations=durations,
    )
    return [
        {"id": index, "files": list(group)}
        for index, group in enumerate(groups, 1)
        if group
    ]


def build_file_lanes(
    paths: list[str],
    *,
    lane_count: int = 4,
    max_jobs_per_lane: int = 256,
) -> list[list[dict[str, object]]]:
    """Build deterministic CI lanes with exactly one test file in every matrix cell.

    GitHub limits a single job matrix to 256 generated jobs. Multiple fixed lanes keep
    each matrix within that bound while preventing an unrelated test file from hiding,
    aborting, or contaminating the causal evidence for another file.
    """

    normalized = sorted({Path(path).as_posix() for path in paths})
    if lane_count < 1:
        raise ValueError("lane_count must be at least 1")
    if max_jobs_per_lane < 1:
        raise ValueError("max_jobs_per_lane must be at least 1")

    capacity = lane_count * max_jobs_per_lane
    if len(normalized) > capacity:
        raise ValueError(
            f"{len(normalized)} test files exceed file-cell capacity {capacity}; "
            "increase the fixed CI lane count before adding more tests"
        )

    lanes: list[list[dict[str, object]]] = [[] for _ in range(lane_count)]
    for global_index, path in enumerate(normalized, 1):
        lane_index = (global_index - 1) % lane_count
        lanes[lane_index].append(
            {
                "id": f"{global_index:04d}",
                "files": [path],
            }
        )

    if any(len(lane) > max_jobs_per_lane for lane in lanes):
        raise AssertionError("file-cell lane exceeds configured GitHub matrix capacity")
    return lanes


def _load_durations(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("duration history must be a JSON object")
    return {str(key): float(value) for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-dir", type=Path, default=Path("tests"))
    parser.add_argument("--max-jobs", type=int, default=256)
    parser.add_argument("--durations", type=Path)
    parser.add_argument(
        "--file-cells",
        action="store_true",
        help="emit one test file per matrix cell instead of duration-balanced groups",
    )
    parser.add_argument("--lane", type=int)
    parser.add_argument("--lane-count", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(str(path) for path in args.tests_dir.glob("test_*.py"))
    if args.file_cells:
        lanes = build_file_lanes(
            paths,
            lane_count=args.lane_count,
            max_jobs_per_lane=args.max_jobs,
        )
        if args.lane is not None:
            if not 1 <= args.lane <= len(lanes):
                raise SystemExit(
                    f"--lane must be between 1 and {len(lanes)}, got {args.lane}"
                )
            payload: object = {"include": lanes[args.lane - 1]}
        else:
            payload = {"lanes": [{"include": lane} for lane in lanes]}
        print(json.dumps(payload, separators=(",", ":")))
        return 0

    matrix = build_matrix(
        paths,
        max_jobs=args.max_jobs,
        durations=_load_durations(args.durations),
    )
    print(json.dumps({"include": matrix}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
