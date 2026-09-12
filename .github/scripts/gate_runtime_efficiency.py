from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finding_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Ignore code motion while retaining the semantic expensive-call identity."""

    return (
        str(row.get("category", "")),
        str(row.get("path", "")),
        str(row.get("function", "")),
        str(row.get("call", "")),
    )


def _duplicate_identity(group: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(copy.get("path", "")), str(copy.get("name", "")))
            for copy in group.get("copies", ())
        )
    )


def _introduced(
    current: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    identity,
) -> list[dict[str, Any]]:
    remaining = Counter(identity(row) for row in baseline)
    introduced: list[dict[str, Any]] = []
    for row in current:
        key = identity(row)
        if remaining[key]:
            remaining[key] -= 1
        else:
            introduced.append(row)
    return introduced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()

    current = _load(args.current)
    baseline = _load(args.baseline)
    introduced_findings = _introduced(
        list(current.get("findings", ())),
        list(baseline.get("findings", ())),
        _finding_identity,
    )
    introduced_duplicates = _introduced(
        list(current.get("exact_duplicate_function_bodies", ())),
        list(baseline.get("exact_duplicate_function_bodies", ())),
        _duplicate_identity,
    )

    print(
        "runtime efficiency delta: "
        f"findings {baseline.get('finding_count', 0)} -> {current.get('finding_count', 0)}, "
        f"new_findings={len(introduced_findings)}, "
        f"new_duplicate_groups={len(introduced_duplicates)}"
    )
    for row in introduced_findings:
        print(
            "  NEW "
            f"{row.get('category')} {row.get('path')}:{row.get('line')} "
            f"{row.get('function')} {row.get('call', '')}"
        )
    for group in introduced_duplicates:
        copies = ", ".join(
            f"{copy.get('path')}:{copy.get('name')}"
            for copy in group.get("copies", ())
        )
        print(f"  NEW exact duplicate function body: {copies}")

    return 1 if introduced_findings or introduced_duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
