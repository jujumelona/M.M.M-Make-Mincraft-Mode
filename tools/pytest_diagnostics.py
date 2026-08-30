from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class JUnitAnalysis:
    total: int
    failed: int
    errors: int
    skipped: int
    groups: tuple[FailureGroup, ...]
    affected: dict[str, list[str]]


def _compact_message(value: str, *, limit: int = 1200) -> str:
    text = " ".join(value.split())
    if not text:
        return "pytest test failed"
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_result_node(case: ET.Element) -> ET.Element | None:
    for child in case:
        if _local_name(child.tag) in {"failure", "error"}:
            return child
    return None


def analyze_junit(path: Path) -> JUnitAnalysis:
    """Parse JUnit once, keeping only failure groups instead of the full XML tree."""

    collector = DiagnosticCollector()
    affected: dict[str, list[str]] = defaultdict(list)
    total = failed = errors = skipped = 0

    for _, element in ET.iterparse(path, events=("end",)):
        if _local_name(element.tag) != "testcase":
            continue
        total += 1
        node = _first_result_node(element)
        has_skipped = any(_local_name(child.tag) == "skipped" for child in element)
        if node is None:
            if has_skipped:
                skipped += 1
            element.clear()
            continue

        node_tag = _local_name(node.tag)
        if node_tag == "failure":
            failed += 1
        else:
            errors += 1
        classname = str(element.attrib.get("classname") or "pytest")
        name = str(element.attrib.get("name") or "unknown")
        nodeid = f"{classname}::{name}"
        message = _compact_message(
            str(node.attrib.get("message") or node.text or "pytest test failed")
        )
        event = FailureEvent(
            stage="ci:pytest",
            operation="test suite",
            category=FailureCategory.VALIDATION,
            cause_type="PytestFailure" if node_tag == "failure" else "PytestError",
            cause=message,
            retryable=False,
            final_status=FailureStatus.FAILED,
        )
        group = collector.record(event)
        if nodeid not in affected[group.event.fingerprint]:
            affected[group.event.fingerprint].append(nodeid)
        element.clear()

    return JUnitAnalysis(
        total=total,
        failed=failed,
        errors=errors,
        skipped=skipped,
        groups=collector.groups(),
        affected=dict(affected),
    )


def failure_groups_from_junit(path: Path) -> tuple[tuple[FailureGroup, ...], dict[str, list[str]]]:
    analysis = analyze_junit(path)
    return analysis.groups, analysis.affected


def _render_analysis_failure_summary(analysis: JUnitAnalysis) -> str:
    if not analysis.groups:
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

    lines = [render_failure_summary(analysis.groups)]
    for index, group in enumerate(analysis.groups[:20], start=1):
        tests = analysis.affected.get(group.event.fingerprint, [])
        shown = tests[:10]
        suffix = f" (+{len(tests) - len(shown)} more)" if len(tests) > len(shown) else ""
        lines.append(f"AFFECTED TESTS {index}\n" + ", ".join(shown) + suffix)
    omitted = len(analysis.groups) - 20
    if omitted > 0:
        lines.append(f"AFFECTED TEST GROUPS OMITTED\n{omitted}")
    return "\n".join(lines)


def render_junit_failure_summary(path: Path) -> str:
    return _render_analysis_failure_summary(analyze_junit(path))


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


def _render_internal_failure(
    *,
    operation: str,
    cause_type: str,
    cause: str,
    fallback: str,
    category: FailureCategory = FailureCategory.INTERNAL,
) -> str:
    collector = DiagnosticCollector()
    collector.record(
        FailureEvent(
            stage="ci:pytest",
            operation=operation,
            category=category,
            cause_type=cause_type,
            cause=cause,
            retryable=False,
            final_status=FailureStatus.FAILED,
            fallback=fallback,
        )
    )
    return render_failure_summary(collector.groups())


def _safe_exit_code(returncode: int) -> int:
    return returncode if 1 <= returncode <= 255 else 1


def _validate_output_paths(log_path: Path, junit_path: Path) -> bool:
    try:
        return log_path.resolve() != junit_path.resolve()
    except OSError:
        return log_path.absolute() != junit_path.absolute()


def _prepare_output_directories(log_path: Path, junit_path: Path) -> bool:
    try:
        for parent in {log_path.parent, junit_path.parent}:
            parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            _render_internal_failure(
                operation="prepare diagnostic output directories",
                cause_type=type(exc).__name__,
                cause=_compact_message(str(exc)),
                fallback=f"log={log_path}; junit={junit_path}",
            )
        )
        return False
    return True


def _remove_stale_outputs(log_path: Path, junit_path: Path) -> bool:
    for path, operation in (
        (log_path, "remove stale pytest log"),
        (junit_path, "remove stale JUnit"),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(
                _render_internal_failure(
                    operation=operation,
                    cause_type=type(exc).__name__,
                    cause=_compact_message(str(exc)),
                    fallback="pytest was not started; no current raw evidence is available",
                )
            )
            return False
    return True


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not _validate_output_paths(args.log, args.junit):
        print(
            _render_internal_failure(
                operation="validate diagnostic outputs",
                cause_type="OutputPathCollision",
                cause="--log and --junit must refer to different files",
                fallback="pytest was not started",
                category=FailureCategory.INPUT,
            )
        )
        return 2

    if not _prepare_output_directories(args.log, args.junit):
        return 1
    if not _remove_stale_outputs(args.log, args.junit):
        return 1

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--maxfail={max(1, args.maxfail)}",
        f"--junitxml={args.junit}",
    ]
    if args.durations > 0:
        command.append(f"--durations={args.durations}")
    command.extend(args.tests)

    try:
        with args.log.open("w", encoding="utf-8", errors="replace") as log_handle:
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except OSError as exc:
        print(
            _render_internal_failure(
                operation="launch pytest",
                cause_type=type(exc).__name__,
                cause=_compact_message(str(exc)),
                fallback=f"raw output target={args.log}",
            )
        )
        return 1

    analysis: JUnitAnalysis | None = None
    if args.junit.is_file():
        try:
            analysis = analyze_junit(args.junit)
        except (OSError, ET.ParseError) as exc:
            print(
                _render_internal_failure(
                    operation="parse JUnit",
                    cause_type=type(exc).__name__,
                    cause=_compact_message(str(exc)),
                    fallback=f"raw pytest output preserved at {args.log}",
                )
            )
            print(f"RAW OUTPUT {args.log}")
            print(f"JUNIT {args.junit}")
            return _safe_exit_code(process.returncode)

    if analysis is None:
        print(
            _render_internal_failure(
                operation="produce JUnit",
                cause_type="MissingJUnit",
                cause="pytest did not produce the requested JUnit report for this run",
                fallback=f"raw pytest output preserved at {args.log}",
            )
        )
        print(f"RAW OUTPUT {args.log}")
        print(f"JUNIT {args.junit}")
        return _safe_exit_code(process.returncode)

    if process.returncode == 0:
        if analysis.failed or analysis.errors:
            print(
                _render_internal_failure(
                    operation="validate pytest/JUnit agreement",
                    cause_type="PytestExitMismatch",
                    cause=(
                        f"pytest exit=0 but JUnit reports failed={analysis.failed} "
                        f"errors={analysis.errors}"
                    ),
                    fallback=f"raw pytest output preserved at {args.log}",
                )
            )
            return 1
        if analysis.total == 0:
            print(
                _render_internal_failure(
                    operation="validate JUnit evidence",
                    cause_type="EmptyJUnit",
                    cause="pytest exited successfully but JUnit contains zero testcases",
                    fallback=f"raw pytest output preserved at {args.log}",
                )
            )
            return 1
        print("FINAL STATUS")
        print("PASS")
        print(
            f"TESTS total={analysis.total} failed={analysis.failed} "
            f"errors={analysis.errors} skipped={analysis.skipped}"
        )
        print(f"RAW OUTPUT {args.log}")
        return 0

    print(_render_analysis_failure_summary(analysis))
    print(
        f"TESTS total={analysis.total} failed={analysis.failed} "
        f"errors={analysis.errors} skipped={analysis.skipped}"
    )
    print(f"RAW OUTPUT {args.log}")
    print(f"JUNIT {args.junit}")
    return _safe_exit_code(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
