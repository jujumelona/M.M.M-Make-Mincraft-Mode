from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
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

if __package__:
    from .audit_stream_redactor import StreamingRedactor
else:
    from audit_stream_redactor import StreamingRedactor

_OUTPUT_CHUNK_CHARS = 64 * 1024
_DEFAULT_TIMEOUT_SECONDS = 2400
_XML_REDACTION_MARKER = "&lt;redacted&gt;"


@dataclass(frozen=True)
class JUnitAnalysis:
    total: int
    failed: int
    errors: int
    skipped: int
    groups: tuple[FailureGroup, ...]
    affected: dict[str, list[str]]


def _environment_secret_values() -> tuple[str, ...]:
    values: list[str] = []
    for name, secret in os.environ.items():
        upper = name.upper()
        if secret and any(
            marker in upper
            for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "COOKIE")
        ):
            values.append(secret)
    return tuple(dict.fromkeys(values))


def _sanitize_text(value: object) -> str:
    redactor = StreamingRedactor(_environment_secret_values())
    return redactor.feed(str(value)) + redactor.finish()


def _compact_message(value: str, *, limit: int = 1200) -> str:
    text = " ".join(_sanitize_text(value).split())
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
        classname = _sanitize_text(str(element.attrib.get("classname") or "pytest"))
        name = _sanitize_text(str(element.attrib.get("name") or "unknown"))
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
        description="Run pytest while preserving redacted raw output and printing compact causal diagnostics."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--maxfail", type=int, default=25)
    parser.add_argument("--durations", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=_DEFAULT_TIMEOUT_SECONDS)
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
            cause=_compact_message(cause),
            retryable=category is FailureCategory.TRANSIENT,
            final_status=FailureStatus.FAILED,
            fallback=_compact_message(fallback),
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
                cause=str(exc),
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
                    cause=str(exc),
                    fallback="pytest was not started; no current raw evidence is available",
                )
            )
            return False
    return True


def _redact_file(source: Path, destination: Path, *, replacement: str = "<redacted>") -> None:
    redactor = StreamingRedactor(
        _environment_secret_values(),
        replacement=replacement,
    )
    with source.open("r", encoding="utf-8", errors="replace") as source_handle, destination.open(
        "w", encoding="utf-8", errors="replace"
    ) as destination_handle:
        while True:
            chunk = source_handle.read(_OUTPUT_CHUNK_CHARS)
            if not chunk:
                break
            safe = redactor.feed(chunk)
            if safe:
                destination_handle.write(safe)
        final = redactor.finish()
        if final:
            destination_handle.write(final)


def _redact_file_in_place(path: Path, *, replacement: str = "<redacted>") -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".redacted.tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        _redact_file(path, temporary_path, replacement=replacement)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _capture_pytest(
    command: list[str],
    log_path: Path,
    *,
    timeout_seconds: int,
) -> tuple[int | None, BaseException | None]:
    temporary_path: Path | None = None
    try:
        log_path.write_text("", encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="replace",
            dir=log_path.parent,
            prefix=".pytest-output-",
            suffix=".log",
            delete=False,
        ) as raw_handle:
            temporary_path = Path(raw_handle.name)
            try:
                process = subprocess.run(
                    command,
                    stdout=raw_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout_seconds,
                )
                returncode: int | None = process.returncode
                failure: BaseException | None = None
            except (OSError, subprocess.TimeoutExpired) as exc:
                returncode = None
                failure = exc
        _redact_file(temporary_path, log_path)
        return returncode, failure
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    if args.timeout_seconds <= 0:
        print(
            _render_internal_failure(
                operation="validate pytest timeout",
                cause_type="InvalidPytestTimeout",
                cause=f"--timeout-seconds must be positive, got {args.timeout_seconds}",
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
        process_returncode, launch_failure = _capture_pytest(
            command,
            args.log,
            timeout_seconds=args.timeout_seconds,
        )
    except OSError as exc:
        print(
            _render_internal_failure(
                operation="preserve redacted pytest output",
                cause_type=type(exc).__name__,
                cause=str(exc),
                fallback=f"redacted output target={args.log}",
            )
        )
        return 1

    if launch_failure is not None:
        if isinstance(launch_failure, subprocess.TimeoutExpired):
            print(
                _render_internal_failure(
                    operation="run pytest",
                    cause_type="TimeoutExpired",
                    cause=f"pytest exceeded timeout={args.timeout_seconds}s",
                    fallback=f"redacted pytest output preserved at {args.log}",
                    category=FailureCategory.TRANSIENT,
                )
            )
            print(f"RAW OUTPUT {args.log}")
            return 124
        print(
            _render_internal_failure(
                operation="launch pytest",
                cause_type=type(launch_failure).__name__,
                cause=str(launch_failure),
                fallback=f"redacted output target={args.log}",
            )
        )
        return 1

    assert process_returncode is not None

    if args.junit.is_file():
        try:
            _redact_file_in_place(args.junit, replacement=_XML_REDACTION_MARKER)
        except OSError as exc:
            print(
                _render_internal_failure(
                    operation="redact JUnit",
                    cause_type=type(exc).__name__,
                    cause=str(exc),
                    fallback=f"redacted pytest output preserved at {args.log}",
                )
            )
            return _safe_exit_code(process_returncode)

    analysis: JUnitAnalysis | None = None
    if args.junit.is_file():
        try:
            analysis = analyze_junit(args.junit)
        except (OSError, ET.ParseError) as exc:
            print(
                _render_internal_failure(
                    operation="parse JUnit",
                    cause_type=type(exc).__name__,
                    cause=str(exc),
                    fallback=f"redacted pytest output preserved at {args.log}",
                )
            )
            print(f"RAW OUTPUT {args.log}")
            print(f"JUNIT {args.junit}")
            return _safe_exit_code(process_returncode)

    if analysis is None:
        print(
            _render_internal_failure(
                operation="produce JUnit",
                cause_type="MissingJUnit",
                cause="pytest did not produce the requested JUnit report for this run",
                fallback=f"redacted pytest output preserved at {args.log}",
            )
        )
        print(f"RAW OUTPUT {args.log}")
        print(f"JUNIT {args.junit}")
        return _safe_exit_code(process_returncode)

    if process_returncode == 0:
        if analysis.failed or analysis.errors:
            print(
                _render_internal_failure(
                    operation="validate pytest/JUnit agreement",
                    cause_type="PytestExitMismatch",
                    cause=(
                        f"pytest exit=0 but JUnit reports failed={analysis.failed} "
                        f"errors={analysis.errors}"
                    ),
                    fallback=f"redacted pytest output preserved at {args.log}",
                )
            )
            return 1
        if analysis.total == 0:
            print(
                _render_internal_failure(
                    operation="validate JUnit evidence",
                    cause_type="EmptyJUnit",
                    cause="pytest exited successfully but JUnit contains zero testcases",
                    fallback=f"redacted pytest output preserved at {args.log}",
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
    return _safe_exit_code(process_returncode)


if __name__ == "__main__":
    raise SystemExit(main())
