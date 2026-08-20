from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable


_DEFAULT_DIAGNOSTIC_PAGE_MAX_FILES = 128
_DEFAULT_DIAGNOSTIC_PAGE_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_DEFAULT_DIAGNOSTIC_QUIET_SECONDS = 2.0


class JDTLanguageServerError(RuntimeError):
    pass


class _JsonRpcProcess:
    def __init__(self, command: list[str], cwd: Path) -> None:
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr: deque[str] = deque(maxlen=30)
        self.workspace_folders = [{"uri": cwd.resolve().as_uri(), "name": cwd.name}]
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._error_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._error_reader.start()
        self._next_id = 1

    def request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        deferred: list[dict[str, Any]] = []
        try:
            while time.monotonic() < deadline:
                try:
                    message = self.messages.get(timeout=min(0.25, deadline - time.monotonic()))
                except queue.Empty:
                    continue
                if _respond_to_server_request(self, message):
                    continue
                if message.get("id") == request_id:
                    if "error" in message:
                        raise JDTLanguageServerError(str(message["error"]))
                    return message.get("result")
                deferred.append(message)
            raise TimeoutError(f"JDT LS request timed out: {method}")
        finally:
            for message in deferred:
                self.messages.put(message)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def send(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise JDTLanguageServerError("JDT LS stdin is unavailable.")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        self.process.stdin.write(body)
        self.process.stdin.flush()

    def close(self) -> None:
        try:
            self.request("shutdown", {}, timeout=5)
        except Exception:
            pass
        try:
            self.notify("exit", {})
        except Exception:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        while True:
            headers: dict[str, str] = {}
            while True:
                line = stream.readline()
                if not line:
                    return
                if line in {b"\r\n", b"\n"}:
                    break
                decoded = line.decode("ascii", errors="replace").strip()
                if ":" in decoded:
                    key, value = decoded.split(":", 1)
                    headers[key.lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            if length <= 0:
                continue
            body = stream.read(length)
            try:
                message = json.loads(body.decode("utf-8"))
            except Exception:
                continue
            if isinstance(message, dict):
                self.messages.put(message)

    def _read_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            self.stderr.append(raw.decode("utf-8", errors="replace").rstrip())


def _respond_to_server_request(
    rpc: _JsonRpcProcess,
    message: dict[str, Any],
) -> bool:
    method = message.get("method")
    if "id" not in message or not isinstance(method, str):
        return False

    params = message.get("params")
    if not isinstance(params, dict):
        params = {}

    if method in {
        "client/registerCapability",
        "client/unregisterCapability",
        "window/showMessageRequest",
        "window/workDoneProgress/create",
    }:
        result: Any = None
    elif method == "workspace/configuration":
        items = params.get("items")
        result = [None] * len(items) if isinstance(items, list) else []
    elif method == "workspace/workspaceFolders":
        result = list(rpc.workspace_folders)
    elif method == "workspace/applyEdit":
        result = {"applied": False}
    else:
        rpc.send(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Unsupported server request: {method}",
                },
            }
        )
        return True

    rpc.send({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
    return True


class JavaLanguageService:
    """Small bounded LSP client for Eclipse JDT LS.

    JDT LS itself requires Java 21; the imported Minecraft project remains Java 17.
    The command is operator configuration, never model-generated input.
    """

    def __init__(
        self,
        command: str | None = None,
        *,
        diagnostic_page_max_files: int = _DEFAULT_DIAGNOSTIC_PAGE_MAX_FILES,
        diagnostic_page_max_source_bytes: int = (
            _DEFAULT_DIAGNOSTIC_PAGE_MAX_SOURCE_BYTES
        ),
        diagnostic_quiet_seconds: float = _DEFAULT_DIAGNOSTIC_QUIET_SECONDS,
    ) -> None:
        raw = command or os.environ.get("MMM_JDTLS_CMD", "").strip()
        self.command = shlex.split(raw) if raw else ["jdtls"]
        if not self.command:
            raise JDTLanguageServerError("Empty JDT LS command.")
        if diagnostic_page_max_files <= 0:
            raise ValueError("diagnostic_page_max_files must be positive.")
        if diagnostic_page_max_source_bytes <= 0:
            raise ValueError(
                "diagnostic_page_max_source_bytes must be positive."
            )
        if diagnostic_quiet_seconds < 0:
            raise ValueError("diagnostic_quiet_seconds cannot be negative.")
        self.diagnostic_page_max_files = diagnostic_page_max_files
        self.diagnostic_page_max_source_bytes = (
            diagnostic_page_max_source_bytes
        )
        self.diagnostic_quiet_seconds = diagnostic_quiet_seconds

    def diagnostics(
        self,
        project_root: str | Path,
        *,
        relative_files: Iterable[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        files = _java_files(root, relative_files)
        pages = _diagnostic_pages(
            files,
            max_files=self.diagnostic_page_max_files,
            max_source_bytes=self.diagnostic_page_max_source_bytes,
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

        rpc = _JsonRpcProcess(self.command, root)
        try:
            rpc.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": root.as_uri(),
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": {"relatedInformation": True}
                        },
                        "workspace": {"workspaceFolders": True},
                    },
                    "workspaceFolders": list(rpc.workspace_folders),
                },
                timeout=min(timeout_seconds, 45),
            )
            rpc.notify("initialized", {})
            diagnostics: dict[str, list[dict[str, Any]]] = {}
            page_receipts: list[dict[str, Any]] = []
            total_source_bytes = 0
            for page_index, page in enumerate(pages):
                sources, source_bytes = _read_source_page(
                    page,
                    max_source_bytes=self.diagnostic_page_max_source_bytes,
                )
                expected_uris = {path.as_uri() for path, _text in sources}
                for path, text in sources:
                    rpc.notify(
                        "textDocument/didOpen",
                        {
                            "textDocument": {
                                "uri": path.as_uri(),
                                "languageId": "java",
                                "version": 1,
                                "text": text,
                            }
                        },
                    )
                page_diagnostics = _collect_diagnostics(
                    rpc,
                    expected_uris=expected_uris,
                    timeout_seconds=timeout_seconds,
                    quiet_seconds=self.diagnostic_quiet_seconds,
                )
                for path, _text in sources:
                    rpc.notify(
                        "textDocument/didClose",
                        {"textDocument": {"uri": path.as_uri()}},
                    )
                diagnostics.update(page_diagnostics)
                page_errors, page_warnings = _diagnostic_counts(
                    page_diagnostics
                )
                relative_paths = [
                    path.relative_to(root).as_posix()
                    for path, _text in sources
                ]
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
        finally:
            rpc.close()

    def workspace_symbols(
        self,
        project_root: str | Path,
        query: str,
        *,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        rpc = _JsonRpcProcess(self.command, root)
        try:
            rpc.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": root.as_uri(),
                    "capabilities": {"workspace": {"symbol": {}}},
                    "workspaceFolders": list(rpc.workspace_folders),
                },
                timeout=min(timeout_seconds, 45),
            )
            rpc.notify("initialized", {})
            result = rpc.request(
                "workspace/symbol",
                {"query": query},
                timeout=timeout_seconds,
            )
            return {
                "schema_version": "mmm/java-symbols-v1",
                "query": query,
                "symbols": result or [],
            }
        finally:
            rpc.close()


def _java_files(root: Path, relative_files: Iterable[str] | None) -> list[Path]:
    if relative_files is None:
        candidates = (
            path.resolve()
            for path in root.rglob("*.java")
            if path.is_file() and not path.is_symlink()
        )
    else:
        requested: list[Path] = []
        for relative in relative_files:
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("Java file escaped the project root.") from exc
            if path.suffix != ".java" or not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            requested.append(path)
        candidates = iter(requested)
    return sorted(set(candidates), key=lambda path: path.as_posix())


def _diagnostic_pages(
    files: Iterable[Path],
    *,
    max_files: int,
    max_source_bytes: int,
) -> list[tuple[Path, ...]]:
    pages: list[tuple[Path, ...]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in files:
        source_bytes = path.stat().st_size
        if source_bytes > max_source_bytes:
            raise ValueError(
                "Java source exceeds the per-page JDT LS source-byte limit: "
                f"{path} ({source_bytes} > {max_source_bytes})."
            )
        if current and (
            len(current) >= max_files
            or current_bytes + source_bytes > max_source_bytes
        ):
            pages.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += source_bytes
    if current:
        pages.append(tuple(current))
    return pages


def _read_source_page(
    page: Iterable[Path],
    *,
    max_source_bytes: int,
) -> tuple[list[tuple[Path, str]], int]:
    sources: list[tuple[Path, str]] = []
    total_bytes = 0
    for path in page:
        raw = path.read_bytes()
        total_bytes += len(raw)
        if total_bytes > max_source_bytes:
            raise ValueError(
                "Java sources changed while preparing a JDT LS page and now "
                "exceed its source-byte limit."
            )
        sources.append((path, raw.decode("utf-8", errors="replace")))
    return sources, total_bytes


def _collect_diagnostics(
    rpc: _JsonRpcProcess,
    *,
    expected_uris: set[str],
    timeout_seconds: float,
    quiet_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    deadline = time.monotonic() + timeout_seconds
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        if time.monotonic() - quiet_since >= quiet_seconds:
            break
        remaining = deadline - time.monotonic()
        wait_seconds = min(
            0.25,
            max(0.001, quiet_seconds),
            max(0.001, remaining),
        )
        try:
            message = rpc.messages.get(timeout=wait_seconds)
        except queue.Empty:
            continue
        if _respond_to_server_request(rpc, message):
            continue
        if message.get("method") != "textDocument/publishDiagnostics":
            continue
        params = message.get("params", {})
        uri = str(params.get("uri", ""))
        values = params.get("diagnostics", [])
        if uri not in expected_uris or not isinstance(values, list):
            continue
        diagnostics[uri] = _sorted_diagnostics(values)
        quiet_since = time.monotonic()
    return dict(sorted(diagnostics.items()))


def _sorted_diagnostics(
    values: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        values,
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )


def _diagnostic_counts(
    diagnostics: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    errors = sum(
        1
        for values in diagnostics.values()
        for item in values
        if int(item.get("severity", 1)) == 1
    )
    warnings = sum(
        1
        for values in diagnostics.values()
        for item in values
        if int(item.get("severity", 2)) == 2
    )
    return errors, warnings


def _diagnostic_result(
    *,
    root: Path,
    files_opened: int,
    total_source_bytes: int,
    page_receipts: list[dict[str, Any]],
    diagnostics: dict[str, list[dict[str, Any]]],
    stderr_tail: list[str],
    max_files: int,
    max_source_bytes: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    deterministic_diagnostics = {
        uri: _sorted_diagnostics(values)
        for uri, values in sorted(diagnostics.items())
    }
    errors, warnings = _diagnostic_counts(deterministic_diagnostics)
    return {
        "schema_version": "mmm/java-diagnostics-v2",
        "project_root": str(root),
        "files_opened": files_opened,
        "total_source_bytes": total_source_bytes,
        "page_count": len(page_receipts),
        "page_limits": {
            "max_files": max_files,
            "max_source_bytes": max_source_bytes,
            "timeout_seconds": timeout_seconds,
        },
        "pages": page_receipts,
        "error_count": errors,
        "warning_count": warnings,
        "diagnostics": deterministic_diagnostics,
        "server_stderr_tail": stderr_tail[-30:],
    }
