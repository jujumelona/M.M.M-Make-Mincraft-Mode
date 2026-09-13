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
    args = parser.parse_args()
    paths = sorted(str(path) for path in args.tests_dir.glob("test_*.py"))
    matrix = build_matrix(
        paths,
        max_jobs=args.max_jobs,
        durations=_load_durations(args.durations),
    )
    print(json.dumps({"include": matrix}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
