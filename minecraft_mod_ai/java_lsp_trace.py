from __future__ import annotations

"""Root-cause-visible JDT LS diagnostics on top of the canonical readiness path.

The traced service intentionally does not implement JDT initialization or readiness.
Those contracts live only in :mod:`minecraft_mod_ai.java_lsp`, so tracing cannot bypass
project-JDK discovery, runtime configuration, or the Object/String semantic probe.
"""

import queue
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .java_lsp import (
    JDTLanguageServerError,
    JavaLanguageService,
    _diagnostic_counts,
    _diagnostic_pages,
    _diagnostic_result,
    _java_files,
    _raise_on_java_core_bootstrap_failure,
    _read_source_page,
    _respond_to_server_request,
    _sorted_diagnostics,
)
from .root_cause_trace import emit_root_cause
from .source_set_boundary_contract import (
    SourceSetBoundaryError,
    assert_server_safe_source_sets,
)


def _record_server_progress(rpc: Any, message: dict[str, Any]) -> None:
    method = str(message.get("method") or "")
    if method not in {
        "language/status",
        "language/eventNotification",
        "$/progress",
        "window/logMessage",
        "window/showMessage",
    }:
        return
    if message.get("_mmm_progress_traced"):
        return
    from .agent_tool_runtime import _sanitize_observation

    entry = {"method": method, "params": _sanitize_observation(message.get("params"))}
    tail = getattr(rpc, "server_progress_tail", None)
    if tail is None:
        tail = deque(maxlen=30)
        rpc.server_progress_tail = tail
    tail.append(entry)
    message["_mmm_progress_traced"] = True
    process = getattr(rpc, "process", None)
    emit_root_cause(
        "jdt_server_progress",
        stage="jdt",
        operation="project_import",
        result="INFO",
        details={"pid": getattr(process, "pid", None), **entry},
    )


class TracedJavaLanguageService(JavaLanguageService):
    """Canonical JavaLanguageService readiness plus root-cause-visible diagnostics."""

    def diagnostics(
        self,
        project_root: str | Path,
        *,
        relative_files: Any = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        try:
            assert_server_safe_source_sets(root)
        except (SourceSetBoundaryError, FileNotFoundError, OSError, UnicodeError) as exc:
            raise JDTLanguageServerError(f"Java source-set preflight failed: {exc}") from exc

        files = _java_files(root, relative_files)
        pages = _diagnostic_pages(
            files,
            max_files=self.diagnostic_page_max_files,
            max_source_bytes=self.diagnostic_page_max_source_bytes,
        )
        emit_root_cause(
            "jdt_diagnostic_input",
            stage="jdt",
            operation="diagnostics",
            gate="source_selection",
            result="PASS",
            details={
                "project_root": str(root),
                "requested_files": list(relative_files) if relative_files is not None else "<all-java-files>",
                "resolved_file_count": len(files),
                "resolved_files": [path.relative_to(root).as_posix() for path in files],
                "page_count": len(pages),
                "timeout_seconds": timeout_seconds,
            },
        )
        if not pages:
            return _diagnostic_result(
                root=root,
                files_opened=0,
                total_source_bytes=0,
                page_receipts=[],
                diagnostics={},
                stderr_tail=[],
                max_files=self.diagnostic_page_max_files,
                max_source_bytes=self.diagnostic_page_max_source_bytes,
                timeout_seconds=timeout_seconds,
            )

        with self._session_lock:
            # Deliberately inherited from JavaLanguageService. This is the single
            # authority for project-JDK discovery, JDT configuration and semantic
            # readiness; ServiceReady alone can never satisfy this boundary.
            rpc = self._ensure_rpc_locked(root, timeout_seconds=timeout_seconds)
            diagnostics: dict[str, list[dict[str, Any]]] = {}
            page_receipts: list[dict[str, Any]] = []
            total_source_bytes = 0
            for page_index, page in enumerate(pages):
                sources, source_bytes = _read_source_page(
                    page,
                    max_source_bytes=self.diagnostic_page_max_source_bytes,
                )
                expected_uris = {path.as_uri() for path, _text in sources}
                relative_paths = [path.relative_to(root).as_posix() for path, _text in sources]
                emit_root_cause(
                    "jdt_did_open_batch",
                    stage="jdt",
                    operation="diagnostics",
                    gate="didOpen",
                    result="START",
                    details={
                        "page_index": page_index,
                        "files": relative_paths,
                        "expected_uris": sorted(expected_uris),
                        "source_bytes": source_bytes,
                        "pid": getattr(getattr(rpc, "process", None), "pid", None),
                    },
                )
                opened_uris: list[str] = []
                try:
                    for source_path, source_text in sources:
                        uri = source_path.as_uri()
                        rpc.notify(
                            "textDocument/didOpen",
                            {
                                "textDocument": {
                                    "uri": uri,
                                    "languageId": "java",
                                    "version": 1,
                                    "text": source_text,
                                }
                            },
                        )
                        opened_uris.append(uri)
                    emit_root_cause(
                        "jdt_did_open_sent",
                        stage="jdt",
                        operation="diagnostics",
                        gate="didOpen",
                        result="PASS",
                        details={"page_index": page_index, "opened_uris": opened_uris},
                    )
                    page_diagnostics = _collect_diagnostics_traced(
                        rpc,
                        expected_uris=expected_uris,
                        timeout_seconds=timeout_seconds,
                        quiet_seconds=self.diagnostic_quiet_seconds,
                        page_index=page_index,
                    )
                    # Core Java type failures prove the JDT workspace itself is not
                    # bootstrapped. They are infrastructure failures, never coder
                    # diagnostics eligible for auto-repair.
                    _raise_on_java_core_bootstrap_failure(page_diagnostics)
                except BaseException as exc:
                    process = getattr(rpc, "process", None)
                    poll = getattr(process, "poll", None)
                    emit_root_cause(
                        "jdt_diagnostic_page_failure",
                        stage="jdt",
                        operation="diagnostics",
                        gate="publishDiagnostics",
                        result="FAIL",
                        reason=f"{type(exc).__name__}: {exc}",
                        details={
                            "page_index": page_index,
                            "expected_uris": sorted(expected_uris),
                            "opened_uris": opened_uris,
                            "returncode": poll() if callable(poll) else None,
                            "reader_alive": bool(getattr(getattr(rpc, "_reader", None), "is_alive", lambda: False)()),
                            "stderr_tail": list(getattr(rpc, "stderr", ()))[-8:],
                            "queued_messages": getattr(getattr(rpc, "messages", None), "qsize", lambda: 0)(),
                            "protocol_counts": dict(getattr(rpc, "protocol_counts", {})),
                            "stdout_eof": bool(getattr(rpc, "stdout_eof", False)),
                        },
                        exc=exc,
                    )
                    raise
                finally:
                    for source_path, _source_text in sources:
                        rpc.notify(
                            "textDocument/didClose",
                            {"textDocument": {"uri": source_path.as_uri()}},
                        )

                diagnostics.update(page_diagnostics)
                page_errors, page_warnings = _diagnostic_counts(page_diagnostics)
                page_receipts.append(
                    {
                        "page_index": page_index,
                        "file_count": len(sources),
                        "source_bytes": source_bytes,
                        "first_file": relative_paths[0],
                        "last_file": relative_paths[-1],
                        "diagnostic_uri_count": len(page_diagnostics),
                        "error_count": page_errors,
                        "warning_count": page_warnings,
                    }
                )
                total_source_bytes += source_bytes

            return _diagnostic_result(
                root=root,
                files_opened=len(files),
                total_source_bytes=total_source_bytes,
                page_receipts=page_receipts,
                diagnostics=diagnostics,
                stderr_tail=list(getattr(rpc, "stderr", ())),
                max_files=self.diagnostic_page_max_files,
                max_source_bytes=self.diagnostic_page_max_source_bytes,
                timeout_seconds=timeout_seconds,
            )


def _collect_diagnostics_traced(
    rpc: Any,
    *,
    expected_uris: set[str],
    timeout_seconds: float,
    quiet_seconds: float,
    page_index: int,
) -> dict[str, list[dict[str, Any]]]:
    if timeout_seconds <= 0:
        raise ValueError("JDT diagnostics timeout must be positive.")
    if quiet_seconds < 0:
        raise ValueError("JDT diagnostics quiet period cannot be negative.")
    if not expected_uris:
        return {}

    diagnostics: dict[str, list[dict[str, Any]]] = {}
    unexpected_uris: set[str] = set()
    ignored_methods: Counter[str] = Counter()
    malformed_messages = 0
    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    settled_since: float | None = None

    while True:
        now = time.monotonic()
        complete = expected_uris.issubset(diagnostics)
        if complete and settled_since is not None and now - settled_since >= quiet_seconds:
            emit_root_cause(
                "jdt_publish_complete",
                stage="jdt",
                operation="diagnostics",
                gate="publishDiagnostics",
                result="PASS",
                details={
                    "page_index": page_index,
                    "observed_uris": sorted(diagnostics),
                    "elapsed_ms": round((now - started) * 1000.0, 3),
                    "ignored_methods": dict(ignored_methods),
                    "unexpected_uris": sorted(unexpected_uris),
                },
            )
            return dict(sorted(diagnostics.items()))

        reader_failure = getattr(rpc, "reader_failure", None) or getattr(
            rpc, "_mmm_reader_failure", None
        )
        if reader_failure is not None:
            raise JDTLanguageServerError(
                "JDT LS stdout reader failed while collecting diagnostics: "
                f"{type(reader_failure).__name__}: {reader_failure}"
            ) from reader_failure

        process = getattr(rpc, "process", None)
        poll = getattr(process, "poll", None)
        returncode = poll() if callable(poll) else None
        if returncode is not None:
            stderr = "\n".join(list(getattr(rpc, "stderr", ()))[-8:])
            raise JDTLanguageServerError(
                "JDT LS exited before publishing complete diagnostics: "
                f"returncode={returncode}; stderr={stderr or '<empty>'}"
            )

        remaining = deadline - now
        if remaining <= 0:
            break
        wait_seconds = min(0.25, remaining)
        if complete and settled_since is not None:
            wait_seconds = min(
                wait_seconds,
                max(0.001, quiet_seconds - (now - settled_since)),
            )
        try:
            message = rpc.messages.get(timeout=max(0.001, wait_seconds))
        except queue.Empty:
            continue
        if _respond_to_server_request(rpc, message):
            ignored_methods[str(message.get("method") or "<server-request>")] += 1
            continue
        if message.get("method") != "textDocument/publishDiagnostics":
            _record_server_progress(rpc, message)
            ignored_methods[str(message.get("method") or "<response>")] += 1
            continue
        params = message.get("params")
        if not isinstance(params, dict):
            malformed_messages += 1
            raise JDTLanguageServerError("JDT LS publishDiagnostics params were not an object.")
        uri = str(params.get("uri") or "")
        if uri not in expected_uris:
            unexpected_uris.add(uri or "<missing-uri>")
            emit_root_cause(
                "jdt_publish_unexpected_uri",
                stage="jdt",
                operation="diagnostics",
                gate="diagnostic_uri_match",
                result="SKIP",
                reason="publishDiagnostics URI did not match an opened Java file",
                details={
                    "page_index": page_index,
                    "uri": uri,
                    "expected_uris": sorted(expected_uris),
                },
            )
            continue
        values = params.get("diagnostics")
        if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
            malformed_messages += 1
            raise JDTLanguageServerError(
                f"JDT LS published malformed diagnostics for expected URI {uri!r}."
            )
        diagnostics[uri] = _sorted_diagnostics(values)
        settled_since = time.monotonic()
        emit_root_cause(
            "jdt_publish_received",
            stage="jdt",
            operation="diagnostics",
            gate="publishDiagnostics",
            result="PASS",
            details={
                "page_index": page_index,
                "uri": uri,
                "diagnostic_count": len(values),
                "observed": len(diagnostics),
                "expected": len(expected_uris),
            },
        )

    missing_uris = sorted(expected_uris.difference(diagnostics))
    state = {
        "page_index": page_index,
        "observed_uris": sorted(diagnostics),
        "missing_uris": missing_uris,
        "unexpected_uris": sorted(unexpected_uris),
        "ignored_methods": dict(ignored_methods),
        "malformed_messages": malformed_messages,
        "process_pid": getattr(process, "pid", None),
        "process_returncode": returncode,
        "reader_alive": bool(getattr(getattr(rpc, "_reader", None), "is_alive", lambda: False)()),
        "stdout_eof": bool(getattr(rpc, "stdout_eof", False)),
        "queued_messages": getattr(getattr(rpc, "messages", None), "qsize", lambda: 0)(),
        "stderr_tail": list(getattr(rpc, "stderr", ()))[-8:],
        "protocol_counts": dict(getattr(rpc, "protocol_counts", {})),
        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
        "server_progress_tail": list(getattr(rpc, "server_progress_tail", ())),
    }
    if missing_uris:
        emit_root_cause(
            "jdt_publish_timeout",
            stage="jdt",
            operation="diagnostics",
            gate="publishDiagnostics",
            result="FAIL",
            reason="JDT LS did not publish diagnostics for every opened Java file",
            details=state,
        )
        raise JDTLanguageServerError(
            "JDT LS did not publish diagnostics for every opened Java file before the validation deadline: "
            f"observed={len(diagnostics)}, expected={len(expected_uris)}, missing={len(missing_uris)}; "
            f"missing_uris={missing_uris}; unexpected_uris={sorted(unexpected_uris)}; "
            f"ignored_methods={dict(ignored_methods)}; process_returncode={returncode}; "
            f"reader_alive={state['reader_alive']}; stdout_eof={state['stdout_eof']}; "
            f"stderr_tail={state['stderr_tail']}"
        )

    emit_root_cause(
        "jdt_quiet_timeout",
        stage="jdt",
        operation="diagnostics",
        gate="diagnostic_quiescence",
        result="FAIL",
        reason="all expected URIs published but diagnostics never reached the quiet period",
        details=state,
    )
    raise JDTLanguageServerError(
        "JDT LS diagnostics did not become quiescent before the validation deadline "
        f"after all {len(expected_uris)} opened Java files were observed; state={state}"
    )


__all__ = ["TracedJavaLanguageService"]
