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

_JAVAC_DIAGNOSTIC = re.compile(
    r"^\s*(?P<path>(?:[A-Za-z]:)?[^:\r\n]+\.java):(?P<line>\d+):\s*"
    r"(?P<kind>error|warning):\s*(?P<message>[^\r\n]*)$",
    re.MULTILINE,
)
_BUILD_LOG_WINDOW_BYTES = 256 * 1024


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
    while "//" in text:
        text = text.replace("//", "/")
    if text.startswith("file://"):
        text = text[7:]
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
    """Extract canonical path-bearing javac diagnostics from build command logs."""

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
        text = bounded_build_log_text(command.get("log_path"))
        for match in _JAVAC_DIAGNOSTIC.finditer(text):
            if len(diagnostics) >= limit:
                break
            body = {
                "path": normalize_source_path(
                    match.group("path"), project_root=project_root
                ),
                "line": int(match.group("line")),
                "severity": 1 if match.group("kind") == "error" else 2,
                "source": "javac",
                "message": match.group("message").strip()[:2000],
                "code": f"javac:{match.group('kind')}:{match.group('line')}",
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


__all__ = [
    "bounded_build_log_text",
    "compiler_log_diagnostics",
    "normalize_source_path",
]
