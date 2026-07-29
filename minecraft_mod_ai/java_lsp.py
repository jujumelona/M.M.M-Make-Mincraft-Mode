from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable


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
        self.stderr: list[str] = []
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


class JavaLanguageService:
    """Small bounded LSP client for Eclipse JDT LS.

    JDT LS itself requires Java 21; the imported Minecraft project remains Java 17.
    The command is operator configuration, never model-generated input.
    """

    def __init__(self, command: str | None = None) -> None:
        raw = command or os.environ.get("MMM_JDTLS_CMD", "").strip()
        self.command = shlex.split(raw) if raw else ["jdtls"]
        if not self.command:
            raise JDTLanguageServerError("Empty JDT LS command.")

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
        files = _java_files(root, relative_files)
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
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                },
                timeout=min(timeout_seconds, 45),
            )
            rpc.notify("initialized", {})
            for path in files:
                rpc.notify(
                    "textDocument/didOpen",
                    {
                        "textDocument": {
                            "uri": path.as_uri(),
                            "languageId": "java",
                            "version": 1,
                            "text": path.read_text(encoding="utf-8", errors="replace"),
                        }
                    },
                )
            diagnostics: dict[str, list[dict[str, Any]]] = {}
            deadline = time.monotonic() + timeout_seconds
            quiet_since = time.monotonic()
            while time.monotonic() < deadline:
                try:
                    message = rpc.messages.get(timeout=0.25)
                except queue.Empty:
                    if time.monotonic() - quiet_since >= 2.0:
                        break
                    continue
                if message.get("method") == "textDocument/publishDiagnostics":
                    params = message.get("params", {})
                    uri = str(params.get("uri", ""))
                    values = params.get("diagnostics", [])
                    if isinstance(values, list):
                        diagnostics[uri] = values
                        quiet_since = time.monotonic()
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
            return {
                "schema_version": "mmm/java-diagnostics-v1",
                "project_root": str(root),
                "files_opened": len(files),
                "error_count": errors,
                "warning_count": warnings,
                "diagnostics": diagnostics,
                "server_stderr_tail": rpc.stderr[-30:],
            }
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
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
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
        paths = sorted(root.rglob("*.java"))
    else:
        paths = []
        for relative in relative_files:
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError("Java file escaped the project root.") from exc
            if path.suffix != ".java" or not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            paths.append(path)
    if len(paths) > 256:
        raise ValueError("JDT LS request is limited to 256 Java files.")
    return paths
