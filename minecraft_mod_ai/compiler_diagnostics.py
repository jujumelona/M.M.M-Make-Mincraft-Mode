from __future__ import annotations

"""Canonical compiler-diagnostic extraction for generation, repair, and feedback.

Gradle/Javac is the source of truth for Java compilation. Every pipeline stage consumes
this same path-normalized shape so a compiler failure can always be bound back to the
generation owner that touched the source.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

_JAVAC_DIAGNOSTIC = re.compile(
    r"^\s*(?P<path>(?:[A-Za-z]:)?[^:\r\n]+\.java):(?P<line>\d+):\s*"
    r"(?P<kind>error|warning):\s*(?P<message>[^\r\n]*)$",
    re.MULTILINE,
)
_BUILD_LOG_WINDOW_BYTES = 256 * 1024
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUNTIME_STACK_FRAME = re.compile(
    r"^\s*at\s+(?:(?:[A-Za-z0-9_.-]+)//)?"
    r"(?P<class>[A-Za-z_$][A-Za-z0-9_$.]*)\.[^\s(]+"
    r"\((?P<file>[A-Za-z0-9_$.-]+\.java):(?P<line>\d+)\)\s*$"
)
_RUNTIME_EXCEPTION = re.compile(
    r"^\s*(?:Caused by:\s*)?"
    r"(?P<type>[A-Za-z_$][A-Za-z0-9_$.]*(?:Exception|Error))"
    r"(?::\s*(?P<message>.*))?\s*$"
)
_MISSING_METHOD_OWNER = re.compile(
    r"(?:^|\s)(?P<class>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+)"
    r"\.[A-Za-z_$<>][\w$<>]*\s*\("
)
_RUNTIME_FRAME_EXCLUDED_PREFIXES = (
    "java.",
    "javax.",
    "jdk.",
    "sun.",
    "org.gradle.",
    "net.fabricmc.",
    "net.minecraft.",
    "com.mojang.",
)


def _sha(value: Mapping[str, Any]) -> str:
    import json

    rendered = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def normalize_source_path(value: Any, *, project_root: str | Path | None = None) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if text.startswith("file:"):
        uri = urlsplit(text)
        text = url2pathname(("//" + uri.netloc if uri.netloc else "") + uri.path)
        text = text.replace("\\", "/")
    if not text:
        return ""

    path = Path(text)
    if project_root is not None:
        try:
            root = Path(project_root).expanduser().resolve()
            resolved = path.expanduser().resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)
            return resolved.relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError):
            pass
    return text.rstrip("/")


def bounded_build_log_text(raw_path: Any) -> str:
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_file() or path.is_symlink():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size <= _BUILD_LOG_WINDOW_BYTES * 2:
                data = stream.read()
            else:
                head = stream.read(_BUILD_LOG_WINDOW_BYTES)
                stream.seek(max(0, size - _BUILD_LOG_WINDOW_BYTES))
                data = head + b"\n" + stream.read(_BUILD_LOG_WINDOW_BYTES)
    except (OSError, ValueError):
        return ""
    return data.decode("utf-8", "replace")


def compiler_log_diagnostics(
    value: Any,
    *,
    project_root: str | Path | None = None,
    limit: int = 256,
) -> list[dict[str, Any]]:
    """Extract canonical source-owned javac and runtime diagnostics from build logs."""

    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()

    def inspect_command(command: Mapping[str, Any]) -> None:
        if len(diagnostics) >= limit:
            return
        exit_code = command.get("exit_code")
        if command.get("timed_out") is not True and (
            not isinstance(exit_code, int) or isinstance(exit_code, bool) or exit_code == 0
        ):
            return
        text = _ANSI_ESCAPE.sub("", bounded_build_log_text(command.get("log_path")))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for match in _JAVAC_DIAGNOSTIC.finditer(text):
            if len(diagnostics) >= limit:
                break
            # Javac's first line often says only "cannot find symbol". Preserve
            # the source/caret and symbol/overload details used by the repairer,
            # without attaching the remainder of the Gradle stack trace.
            details = []
            following = text[match.end():].lstrip("\n").splitlines()[:16]
            if len(following) >= 2 and re.fullmatch(r"\s*\^+\s*", following[1]):
                details.extend(line.strip() for line in following[:2])
                following = following[2:]
            for line in following:
                if re.match(r"\s*(symbol|location|required|found|reason|where):", line) or details and line.startswith("    ") and line.strip() and not (
                    _JAVAC_DIAGNOSTIC.match(line) or line.lstrip().startswith((">", "at "))
                ):
                    details.append(line.strip())
                else:
                    break
            message = "\n".join([match.group("message").strip(), *details])
            body = {
                "path": normalize_source_path(
                    match.group("path"), project_root=project_root
                ),
                "line": int(match.group("line")),
                "severity": 1 if match.group("kind") == "error" else 2,
                "source": "javac",
                "message": message[:2000],
                "code": f"javac:{match.group('kind')}:{match.group('line')}",
            }
            fingerprint = _sha(body)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            diagnostics.append({**body, "diagnostic_sha256": fingerprint})

        runtime_message = ""
        for raw_line in text.splitlines():
            exception = _RUNTIME_EXCEPTION.match(raw_line)
            if exception is not None:
                detail = str(exception.group("message") or "").strip()
                runtime_message = str(exception.group("type"))
                if detail:
                    runtime_message += ": " + detail
                if exception.group("type") == "java.lang.NoSuchMethodError":
                    missing = _MISSING_METHOD_OWNER.search(detail.strip("'\""))
                    if missing is not None:
                        owner = missing.group("class").split("$", 1)[0]
                        if not owner.startswith(_RUNTIME_FRAME_EXCLUDED_PREFIXES):
                            body = {
                                "path": "src/main/java/" + owner.replace(".", "/") + ".java",
                                "severity": 1, "source": "runtime",
                                "message": runtime_message,
                                "code": "runtime:linkage:NoSuchMethodError",
                            }
                            fingerprint = _sha(body)
                            if fingerprint not in seen and len(diagnostics) < limit:
                                seen.add(fingerprint)
                                diagnostics.append({**body, "diagnostic_sha256": fingerprint})
                continue

            frame = _RUNTIME_STACK_FRAME.match(raw_line)
            if frame is None or len(diagnostics) >= limit:
                continue
            class_name = str(frame.group("class") or "")
            if class_name.startswith(_RUNTIME_FRAME_EXCLUDED_PREFIXES):
                continue
            source_class = class_name.split("$", 1)[0]
            if "." not in source_class:
                continue
            source_path = "src/main/java/" + source_class.replace(".", "/") + ".java"
            body = {
                "path": normalize_source_path(
                    source_path, project_root=project_root
                ),
                "line": int(frame.group("line")),
                "severity": 1,
                "source": "runtime",
                "message": runtime_message or "runtime exception reached generated source",
                "code": f"runtime:stack:{frame.group('line')}",
            }
            fingerprint = _sha(body)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            diagnostics.append({**body, "diagnostic_sha256": fingerprint})

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 10 or len(diagnostics) >= limit:
            return
        if isinstance(node, Mapping):
            commands = node.get("commands")
            if isinstance(commands, Sequence) and not isinstance(
                commands, (str, bytes, bytearray)
            ):
                for command in commands:
                    if isinstance(command, Mapping):
                        inspect_command(command)
            for child in node.values():
                if isinstance(child, (Mapping, list, tuple)):
                    walk(child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child, depth + 1)

    walk(value)
    return diagnostics



def gradle_failure_summary(value: Mapping[str, Any]) -> str:
    """Bounded, credential-redacted first-failure evidence for terminal errors.

    This is for human diagnostics only. Source ownership and retry decisions
    continue to use the structured build receipt and compiler diagnostics.
    """
    from .agent_tool_runtime import _redact_text

    fragments: list[str] = []
    error = str(value.get("error") or "").strip()
    if error:
        fragments.append(error[:160])
    commands = value.get("commands")
    failed: Mapping[str, Any] | None = None
    if isinstance(commands, (list, tuple)):
        for command in reversed(commands):
            if not isinstance(command, Mapping):
                continue
            code = command.get("exit_code")
            if command.get("timed_out") is True or (
                isinstance(code, int) and not isinstance(code, bool) and code != 0
            ):
                failed = command
                break
    if failed is not None:
        fragments.append(
            f"command={str(failed.get('name') or 'gradle')[:60]}"
            f" exit={str(failed.get('exit_code'))[:16]}"
            f" log={str(failed.get('log_path') or '')[:350]}"
        )
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, (list, tuple)):
        compiler_errors_reported = 0
        for item in diagnostics:
            if not isinstance(item, Mapping):
                continue
            if item.get("severity") not in {1, "error", "ERROR"}:
                continue
            path = str(item.get("path") or "")[:240]
            line = str(item.get("line") or "?")[:16]
            # Preserve javac's symbol/location detail. Showing only
            # "cannot find symbol" forces another 20-minute compile cycle
            # merely to discover which API signature was actually wrong.
            detail_lines = [
                line.strip() for line in str(item.get("message") or "").splitlines()
                if line.strip()
            ][:5]
            message = " | ".join(detail_lines)[:420]
            fragments.append(f"diagnostic={path}:{line}: {message}")
            compiler_errors_reported += 1
            if compiler_errors_reported >= 8:
                break
    if failed is not None:
        log = _ANSI_ESCAPE.sub("", bounded_build_log_text(failed.get("log_path")))
        if failed.get("timed_out") is True:
            # GameTest may hang without returning an exit code or XML report.
            # Surface the *actual* last progress markers from the server log
            # rather than presenting the timeout as a code-generation defect.
            lines = [
                line.strip() for line in log.splitlines()
                if line.strip() and "M.M.M Make Mincraft Mode: command timed out" not in line
            ]
            if lines:
                fragments.append("last_verifier_output=" + " | ".join(lines[-8:])[:700])
            else:
                fragments.append("last_verifier_output=NO_LOG_OUTPUT")
        match = re.search(r"(?m)^\s*\* What went wrong:\s*$", log)
        if match is not None:
            lines: list[str] = []
            for raw in log[match.end():].splitlines():
                line = raw.strip()
                if line.startswith(("* Try:", "* Exception is:", "* Get more help")):
                    break
                if line:
                    lines.append(line)
                if len(lines) >= 4:
                    break
            if lines:
                fragments.append("gradle_cause=" + " | ".join(lines)[:440])
    # Changed 2026-10-08: show several independent errors from one failed
    # compile so a user need not repeat an entire Colab run to reveal error #2.
    return _redact_text("; ".join(fragments))[:4000]


__all__ = [
    "bounded_build_log_text",
    "gradle_failure_summary",
    "compiler_log_diagnostics",
    "normalize_source_path",
]
