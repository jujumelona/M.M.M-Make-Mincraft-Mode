from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from minecraft_mod_ai.diagnostics import (
    DiagnosticCollector,
    FailureCategory,
    FailureEvent,
    FailureGroup,
    FailureStatus,
    render_failure_summary,
)


def _compact_message(value: str, *, limit: int = 1200) -> str:
    text = " ".join(value.split())
    return text[:limit] if text else "pytest test failed"


def failure_groups_from_junit(path: Path) -> tuple[tuple[FailureGroup, ...], dict[str, list[str]]]:
    root = ET.parse(path).getroot()
    collector = DiagnosticCollector()
    affected: dict[str, list[str]] = defaultdict(list)
    for case in root.iter("testcase"):
        node = case.find("failure")
        if node is None:
            node = case.find("error")
        if node is None:
            continue
        classname = str(case.attrib.get("classname") or "pytest")
        name = str(case.attrib.get("name") or "unknown")
        nodeid = f"{classname}::{name}"
        message = _compact_message(str(node.attrib.get("message") or node.text or "pytest test failed"))
        event = FailureEvent(
            stage="ci:pytest",
            operation="test suite",
            category=FailureCategory.VALIDATION,
            cause_type="PytestFailure" if node.tag == "failure" else "PytestError",
            cause=message,
            retryable=False,
            final_status=FailureStatus.FAILED,
        )
        group = collector.record(event)
        if nodeid not in affected[group.event.fingerprint]:
            affected[group.event.fingerprint].append(nodeid)
    return collector.groups(), dict(affected)


def render_junit_failure_summary(path: Path) -> str:
    groups, affected = failure_groups_from_junit(path)
    if not groups:
        collector = DiagnosticCollector()
        collector.record(
            FailureEvent(
                stage="ci:pytest",
                operation="test suite",
                category=FailureCategory.INTERNAL,
                cause_type="MissingFailureNode",
                cause="pytest exited unsuccessfully but JUnit contained no failure/error nodes",
                retryable=False,
                final_status=FailureStatus.FAILED,
            )
        )
        return render_failure_summary(collector.groups())
    lines = [render_failure_summary(groups)]
    for index, group in enumerate(groups, start=1):
        tests = affected.get(group.event.fingerprint, [])
        shown = tests[:10]
        suffix = f" (+{len(tests) - len(shown)} more)" if len(tests) > len(shown) else ""
        lines.append(f"AFFECTED TESTS {index}\n" + ", ".join(shown) + suffix)
    return "\n".join(lines)


def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = sum(1 for case in cases if case.find("failure") is not None)
    errors = sum(1 for case in cases if case.find("error") is not None)
    skipped = sum(1 for case in cases if case.find("skipped") is not None)
    return len(cases), failed, errors, skipped


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest while preserving raw output as an artifact and printing compact causal diagnostics."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--maxfail", type=int, default=25)
    parser.add_argument("--durations", type=int, default=15)
    parser.add_argument("tests", nargs="+")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.junit.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--maxfail={max(1, args.maxfail)}",
        f"--durations={max(0, args.durations)}",
        f"--junitxml={args.junit}",
        *args.tests,
    ]
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    args.log.write_text(process.stdout or "", encoding="utf-8")
    if process.returncode == 0:
        try:
            total, failed, errors, skipped = _junit_counts(args.junit)
        except (OSError, ET.ParseError):
            total = failed = errors = skipped = 0
        print("FINAL STATUS")
        print("PASS")
        print(f"TESTS total={total} failed={failed} errors={errors} skipped={skipped}")
        print(f"RAW OUTPUT {args.log}")
        return 0
    if args.junit.is_file():
        try:
            print(render_junit_failure_summary(args.junit))
        except (OSError, ET.ParseError) as exc:
            print(
                "ROOT FAILURE 1\n"
                "ci:pytest / parse JUnit [INTERNAL]\n"
                f"CAUSE\n{type(exc).__name__}: {_compact_message(str(exc))}\n"
                "ATTEMPTS\n1\nFALLBACK\nraw pytest output preserved\n"
                "FINAL STATUS\nFAILED"
            )
    else:
        print(
            "ROOT FAILURE 1\n"
            "ci:pytest / produce JUnit [INTERNAL]\n"
            "CAUSE\nMissingJUnit: pytest did not produce the requested JUnit report\n"
            "ATTEMPTS\n1\nFALLBACK\nraw pytest output preserved\n"
            "FINAL STATUS\nFAILED"
        )
    print(f"RAW OUTPUT {args.log}")
    print(f"JUNIT {args.junit}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
