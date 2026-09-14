from __future__ import annotations

import argparse
import os
import signal
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
_DEFAULT_FAULTHANDLER_TIMEOUT_SECONDS = 300
_PROCESS_EXIT_GRACE_SECONDS = 5
_PROCESS_SNAPSHOT_TIMEOUT_SECONDS = 15
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
    parser.add_argument(
        "--faulthandler-timeout-seconds",
        type=int,
        default=_DEFAULT_FAULTHANDLER_TIMEOUT_SECONDS,
        help="Dump all Python thread stacks if a single pytest process makes no progress for this many seconds.",
    )
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


def _redact_file(
    source: Path,
    destination: Path,
    *,
    replacement: str = "<redacted>",
    preserve_xml_declaration: bool = False,
) -> None:
    redactor = StreamingRedactor(
        _environment_secret_values(),
        replacement=replacement,
    )
    with source.open("r", encoding="utf-8", errors="replace") as source_handle, destination.open(
        "w", encoding="utf-8", errors="replace"
    ) as destination_handle:
        chunk = source_handle.read(_OUTPUT_CHUNK_CHARS)
        if preserve_xml_declaration and chunk.startswith("<?xml"):
            declaration_end = chunk.find("?>")
            if declaration_end >= 0:
                declaration_end += 2
                destination_handle.write(chunk[:declaration_end])
                chunk = chunk[declaration_end:]

        while chunk:
            safe = redactor.feed(chunk)
            if safe:
                destination_handle.write(safe)
            chunk = source_handle.read(_OUTPUT_CHUNK_CHARS)

        final = redactor.finish()
        if final:
            destination_handle.write(final)


def _redact_file_in_place(
    path: Path,
    *,
    replacement: str = "<redacted>",
    preserve_xml_declaration: bool = False,
) -> None:
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
        _redact_file(
            path,
            temporary_path,
            replacement=replacement,
            preserve_xml_declaration=preserve_xml_declaration,
        )
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _popen_group_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _snapshot_command() -> list[str]:
    if os.name == "nt":
        return ["tasklist", "/V"]
    return ["ps", "-eo", "pid,ppid,pgid,sid,stat,etime,pcpu,pmem,args", "--forest"]


def _append_process_snapshot(raw_handle: object, *, pytest_pid: int) -> None:
    raw_handle.write("\n\n=== PYTEST TIMEOUT PROCESS SNAPSHOT ===\n")
    raw_handle.write(f"pytest_pid={pytest_pid} platform={sys.platform} os_name={os.name}\n")
    command = _snapshot_command()
    try:
        snapshot = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            errors="replace",
            timeout=_PROCESS_SNAPSHOT_TIMEOUT_SECONDS,
        )
        raw_handle.write(f"$ {' '.join(command)}\n")
        raw_handle.write(snapshot.stdout or "<no process snapshot output>\n")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw_handle.write(f"snapshot command failed: {command!r}: {type(exc).__name__}: {exc}\n")
    raw_handle.flush()


def _terminate_windows_tree(process: subprocess.Popen[object]) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_PROCESS_SNAPSHOT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
    try:
        process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_posix_tree(process: subprocess.Popen[object]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()
    process.wait()


def _terminate_process_tree(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        _terminate_windows_tree(process)
    else:
        _terminate_posix_tree(process)


def _launch_and_wait(
    command: list[str],
    raw_handle: object,
    *,
    timeout_seconds: int,
) -> tuple[int | None, BaseException | None]:
    process: subprocess.Popen[object] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=raw_handle,
            stderr=subprocess.STDOUT,
            **_popen_group_kwargs(),
        )
        return process.wait(timeout=timeout_seconds), None
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            _append_process_snapshot(raw_handle, pytest_pid=process.pid)
            _terminate_process_tree(process)
        return None, exc
    except OSError as exc:
        if process is not None:
            _terminate_process_tree(process)
        return None, exc


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
            outcome = _launch_and_wait(command, raw_handle, timeout_seconds=timeout_seconds)
        _redact_file(temporary_path, log_path)
        return outcome
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _invalid_argument(
    *, operation: str, cause_type: str, cause: str, fallback: str = "pytest was not started"
) -> int:
    print(
        _render_internal_failure(
            operation=operation,
            cause_type=cause_type,
            cause=cause,
            fallback=fallback,
            category=FailureCategory.INPUT,
        )
    )
    return 2


def _validate_args(args: argparse.Namespace) -> int | None:
    if not _validate_output_paths(args.log, args.junit):
        return _invalid_argument(
            operation="validate diagnostic outputs",
            cause_type="OutputPathCollision",
            cause="--log and --junit must refer to different files",
        )
    if args.timeout_seconds <= 0:
        return _invalid_argument(
            operation="validate pytest timeout",
            cause_type="InvalidPytestTimeout",
            cause=f"--timeout-seconds must be positive, got {args.timeout_seconds}",
        )
    if args.faulthandler_timeout_seconds <= 0:
        return _invalid_argument(
            operation="validate faulthandler timeout",
            cause_type="InvalidFaulthandlerTimeout",
            cause=(
                "--faulthandler-timeout-seconds must be positive, got "
                f"{args.faulthandler_timeout_seconds}"
            ),
        )
    return None


def _build_pytest_command(args: argparse.Namespace) -> list[str]:
    stack_timeout_seconds = min(
        args.faulthandler_timeout_seconds,
        max(1, args.timeout_seconds - 1),
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--maxfail={max(1, args.maxfail)}",
        f"--junitxml={args.junit}",
        "-o",
        f"faulthandler_timeout={stack_timeout_seconds}",
    ]
    if args.durations > 0:
        command.append(f"--durations={args.durations}")
    command.extend(args.tests)
    return command


def _handle_launch_failure(failure: BaseException | None, args: argparse.Namespace) -> int | None:
    if failure is None:
        return None
    if isinstance(failure, subprocess.TimeoutExpired):
        print(
            _render_internal_failure(
                operation="run pytest",
                cause_type="TimeoutExpired",
                cause=f"pytest exceeded timeout={args.timeout_seconds}s",
                fallback=(
                    f"redacted pytest output, thread dumps, and process snapshot preserved at {args.log}"
                ),
                category=FailureCategory.TRANSIENT,
            )
        )
        print(f"RAW OUTPUT {args.log}")
        return 124
    print(
        _render_internal_failure(
            operation="launch pytest",
            cause_type=type(failure).__name__,
            cause=str(failure),
            fallback=f"redacted output target={args.log}",
        )
    )
    return 1


def _redact_junit(args: argparse.Namespace, process_returncode: int) -> int | None:
    if not args.junit.is_file():
        return None
    try:
        _redact_file_in_place(
            args.junit,
            replacement=_XML_REDACTION_MARKER,
            preserve_xml_declaration=True,
        )
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
    return None


def _read_analysis(args: argparse.Namespace, process_returncode: int) -> tuple[JUnitAnalysis | None, int | None]:
    if not args.junit.is_file():
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
        return None, _safe_exit_code(process_returncode)
    try:
        return analyze_junit(args.junit), None
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
        return None, _safe_exit_code(process_returncode)


def _report_success(args: argparse.Namespace, analysis: JUnitAnalysis) -> int:
    if analysis.failed or analysis.errors:
        print(
            _render_internal_failure(
                operation="validate pytest/JUnit agreement",
                cause_type="PytestExitMismatch",
                cause=(
                    f"pytest exit=0 but JUnit reports failed={analysis.failed} errors={analysis.errors}"
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


def _report_failure(args: argparse.Namespace, returncode: int, analysis: JUnitAnalysis) -> int:
    print(_render_analysis_failure_summary(analysis))
    print(
        f"TESTS total={analysis.total} failed={analysis.failed} "
        f"errors={analysis.errors} skipped={analysis.skipped}"
    )
    print(f"RAW OUTPUT {args.log}")
    print(f"JUNIT {args.junit}")
    return _safe_exit_code(returncode)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    invalid = _validate_args(args)
    if invalid is not None:
        return invalid
    if not _prepare_output_directories(args.log, args.junit):
        return 1
    if not _remove_stale_outputs(args.log, args.junit):
        return 1

    try:
        process_returncode, launch_failure = _capture_pytest(
            _build_pytest_command(args),
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

    launch_exit = _handle_launch_failure(launch_failure, args)
    if launch_exit is not None:
        return launch_exit
    assert process_returncode is not None

    redact_exit = _redact_junit(args, process_returncode)
    if redact_exit is not None:
        return redact_exit
    analysis, analysis_exit = _read_analysis(args, process_returncode)
    if analysis_exit is not None:
        return analysis_exit
    assert analysis is not None
    if process_returncode == 0:
        return _report_success(args, analysis)
    return _report_failure(args, process_returncode, analysis)


if __name__ == "__main__":
    raise SystemExit(main())