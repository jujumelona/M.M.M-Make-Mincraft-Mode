from __future__ import annotations

"""Root-cause-visible JDT LS client.

This keeps the production JDT contract intact while making process, protocol,
initialization, didOpen and publishDiagnostics failures observable instead of
collapsing them into an anonymous diagnostics-publication timeout.
"""

import json
import queue
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .java_lsp import (
    JDTLanguageServerError,
    JavaLanguageService,
    _JsonRpcProcess,
    _diagnostic_counts,
    _diagnostic_pages,
    _diagnostic_result,
    _read_source_page,
    _respond_to_server_request,
    _sorted_diagnostics,
)
from .root_cause_trace import emit_root_cause


class _TracedJsonRpcProcess(_JsonRpcProcess):
    def __init__(self, command: list[str], cwd: Path) -> None:
        self.reader_failure: BaseException | None = None
        self.stdout_eof = False
        self.protocol_counts: Counter[str] = Counter()
        super().__init__(command, cwd)
        emit_root_cause(
            "jdt_process_started",
            stage="jdt",
            operation="spawn",
            gate="process_start",
            result="PASS",
            details={
                "pid": getattr(self.process, "pid", None),
                "cwd": str(cwd),
                "command": command,
            },
        )

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            failure = JDTLanguageServerError("JDT LS stdout is unavailable.")
            self.reader_failure = failure
            emit_root_cause(
                "jdt_stdout_failure",
                stage="jdt",
                operation="stdout_reader",
                gate="stdout_available",
                result="FAIL",
                reason=str(failure),
                exc=failure,
            )
            return
        try:
            while True:
                headers: dict[str, str] = {}
                while True:
                    line = stream.readline()
                    if not line:
                        self.stdout_eof = True
                        emit_root_cause(
                            "jdt_stdout_eof",
                            stage="jdt",
                            operation="stdout_reader",
                            gate="protocol_stream",
                            result="FAIL" if self.process.poll() is None else "SKIP",
                            reason="JDT LS stdout reached EOF",
                            details={"returncode": self.process.poll()},
                        )
                        return
                    if line in {b"\r\n", b"\n"}:
                        break
                    decoded = line.decode("ascii", errors="replace").strip()
                    if ":" in decoded:
                        key, value = decoded.split(":", 1)
                        headers[key.lower()] = value.strip()
                raw_length = headers.get("content-length", "")
                try:
                    length = int(raw_length or "0")
                except ValueError as exc:
                    raise JDTLanguageServerError(
                        f"JDT LS emitted invalid Content-Length header: {raw_length!r}"
                    ) from exc
                if length <= 0:
                    raise JDTLanguageServerError(
                        f"JDT LS emitted a protocol frame without positive Content-Length: {raw_length!r}"
                    )
                body = stream.read(length)
                if len(body) != length:
                    raise JDTLanguageServerError(
                        "JDT LS stdout ended inside a JSON-RPC body: "
                        f"expected_bytes={length}, observed_bytes={len(body)}"
                    )
                try:
                    message = json.loads(body.decode("utf-8"))
                except Exception as exc:
                    raise JDTLanguageServerError(
                        "JDT LS emitted malformed JSON-RPC JSON: "
                        f"bytes={len(body)}"
                    ) from exc
                if not isinstance(message, dict):
                    raise JDTLanguageServerError(
                        f"JDT LS emitted non-object JSON-RPC payload: {type(message).__name__}"
                    )
                method = str(message.get("method") or "<response>")
                self.protocol_counts[method] += 1
                self.messages.put(message)
        except BaseException as exc:
            self.reader_failure = exc
            setattr(self, "_mmm_reader_failure", exc)
            emit_root_cause(
                "jdt_stdout_failure",
                stage="jdt",
                operation="stdout_reader",
                gate="json_rpc_decode",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={
                    "returncode": self.process.poll(),
                    "protocol_counts": dict(self.protocol_counts),
                    "stderr_tail": list(self.stderr)[-8:],
                },
                exc=exc,
            )


class TracedJavaLanguageService(JavaLanguageService):
    """JavaLanguageService with evidence for every JDT lifecycle boundary."""

    def _ensure_rpc_locked(
        self,
        root: Path,
        *,
        timeout_seconds: int,
    ) -> _JsonRpcProcess:
        rpc = self._rpc
        if rpc is not None and self._project_root == root and self._rpc_alive(rpc):
            emit_root_cause(
                "jdt_session_reused",
                stage="jdt",
                operation="initialize",
                gate="session_health",
                result="PASS",
                details={"project_root": str(root), "pid": getattr(rpc.process, "pid", None)},
            )
            return rpc
        if rpc is not None:
            emit_root_cause(
                "jdt_session_replaced",
                stage="jdt",
                operation="initialize",
                gate="session_health",
                result="SKIP",
                reason="existing JDT session is for another root or is no longer alive",
                details={"old_root": str(self._project_root or ""), "new_root": str(root), "returncode": rpc.process.poll()},
            )
            self._close_rpc_locked()

        emit_root_cause(
            "jdt_initialize_start",
            stage="jdt",
            operation="initialize",
            gate="lsp_initialize",
            result="START",
            details={"project_root": str(root), "timeout_seconds": min(timeout_seconds, 45)},
        )
        traced_rpc = _TracedJsonRpcProcess(self.command, root)
        try:
            initialize_result = traced_rpc.request(
                "initialize",
                {
                    "processId": __import__("os").getpid(),
                    "rootUri": root.as_uri(),
                    "capabilities": {
                        "textDocument": {"publishDiagnostics": {"relatedInformation": True}},
                        "workspace": {"workspaceFolders": True, "symbol": {}},
                    },
                    "workspaceFolders": list(traced_rpc.workspace_folders),
                },
                timeout=min(timeout_seconds, 45),
            )
            traced_rpc.notify("initialized", {})
        except BaseException as exc:
            emit_root_cause(
                "jdt_initialize_failure",
                stage="jdt",
                operation="initialize",
                gate="lsp_initialize",
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                details={
                    "pid": getattr(traced_rpc.process, "pid", None),
                    "returncode": traced_rpc.process.poll(),
                    "stderr_tail": list(traced_rpc.stderr)[-8:],
                    "protocol_counts": dict(traced_rpc.protocol_counts),
                },
                exc=exc,
            )
            traced_rpc.close()
            raise
        emit_root_cause(
            "jdt_initialize_success",
            stage="jdt",
            operation="initialize",
            gate="lsp_initialize",
            result="PASS",
            details={
                "pid": getattr(traced_rpc.process, "pid", None),
                "server_capability_keys": sorted(initialize_result.get("capabilities", {}).keys()) if isinstance(initialize_result, dict) and isinstance(initialize_result.get("capabilities"), dict) else [],
            },
        )
        self._rpc = traced_rpc
        self._project_root = root
        return traced_rpc

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

        from .java_lsp import _java_files

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
                        "pid": getattr(rpc.process, "pid", None),
                    },
                )
                opened_uris: list[str] = []
                try:
                    for source_path, source_text in sources:
                        uri = source_path.as_uri()
                        rpc.notify(
                            "textDocument/didOpen",
                            {"textDocument": {"uri": uri, "languageId": "java", "version": 1, "text": source_text}},
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
                except BaseException as exc:
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
                            "returncode": rpc.process.poll(),
                            "reader_alive": rpc._reader.is_alive(),
                            "stderr_tail": list(rpc.stderr)[-8:],
                            "queued_messages": rpc.messages.qsize(),
                            "protocol_counts": dict(getattr(rpc, "protocol_counts", {})),
                            "stdout_eof": bool(getattr(rpc, "stdout_eof", False)),
                        },
                        exc=exc,
                    )
                    raise
                finally:
                    for source_path, _source_text in sources:
                        rpc.notify("textDocument/didClose", {"textDocument": {"uri": source_path.as_uri()}})

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
                stderr_tail=list(rpc.stderr),
                max_files=self.diagnostic_page_max_files,
                max_source_bytes=self.diagnostic_page_max_source_bytes,
                timeout_seconds=timeout_seconds,
            )


def _collect_diagnostics_traced(
    rpc: _JsonRpcProcess,
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

        reader_failure = getattr(rpc, "reader_failure", None) or getattr(rpc, "_mmm_reader_failure", None)
        if reader_failure is not None:
            raise JDTLanguageServerError(
                "JDT LS stdout reader failed while collecting diagnostics: "
                f"{type(reader_failure).__name__}: {reader_failure}"
            ) from reader_failure

        returncode = rpc.process.poll()
        if returncode is not None:
            stderr = "\n".join(list(rpc.stderr)[-8:])
            raise JDTLanguageServerError(
                "JDT LS exited before publishing complete diagnostics: "
                f"returncode={returncode}; stderr={stderr or '<empty>'}"
            )

        remaining = deadline - now
        if remaining <= 0:
            break
        wait_seconds = min(0.25, remaining)
        if complete and settled_since is not None:
            wait_seconds = min(wait_seconds, max(0.001, quiet_seconds - (now - settled_since)))
        try:
            message = rpc.messages.get(timeout=max(0.001, wait_seconds))
        except queue.Empty:
            continue
        if _respond_to_server_request(rpc, message):
            ignored_methods[str(message.get("method") or "<server-request>")] += 1
            continue
        if message.get("method") != "textDocument/publishDiagnostics":
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
                details={"page_index": page_index, "uri": uri, "expected_uris": sorted(expected_uris)},
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
        "process_pid": getattr(rpc.process, "pid", None),
        "process_returncode": rpc.process.poll(),
        "reader_alive": rpc._reader.is_alive(),
        "stdout_eof": bool(getattr(rpc, "stdout_eof", False)),
        "queued_messages": rpc.messages.qsize(),
        "stderr_tail": list(rpc.stderr)[-8:],
        "protocol_counts": dict(getattr(rpc, "protocol_counts", {})),
        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
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
            f"ignored_methods={dict(ignored_methods)}; process_returncode={rpc.process.poll()}; "
            f"reader_alive={rpc._reader.is_alive()}; stdout_eof={bool(getattr(rpc, 'stdout_eof', False))}; "
            f"stderr_tail={list(rpc.stderr)[-8:]}"
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
