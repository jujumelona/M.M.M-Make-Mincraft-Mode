from __future__ import annotations

import subprocess
import threading
import time
from functools import wraps
from typing import Any


def install(java_lsp_module: Any) -> None:
    """Harden the private JDT JSON-RPC subprocess against silent reader death.

    The legacy client could leave a killed JDT process unreaped and could wait until
    the full request timeout after its stdout reader crashed (for example on a
    malformed Content-Length header). Requests are also serialized so two callers
    cannot repeatedly dequeue and defer one another's responses from the shared queue.
    Diagnostic collection additionally fails closed unless every opened Java URI has
    published one diagnostics notification, including an explicit empty list for a
    clean source file.
    """

    cls = java_lsp_module._JsonRpcProcess

    current_collect = java_lsp_module._collect_diagnostics
    if not getattr(current_collect, "_mmm_jdt_complete_diagnostics", False):

        @wraps(current_collect)
        def collect_diagnostics(
            rpc: Any,
            *,
            expected_uris: set[str],
            timeout_seconds: float,
            quiet_seconds: float,
        ) -> dict[str, list[dict[str, Any]]]:
            if timeout_seconds <= 0:
                raise ValueError("JDT diagnostics timeout must be positive.")
            if quiet_seconds < 0:
                raise ValueError("JDT diagnostics quiet period cannot be negative.")
            if not expected_uris:
                return {}

            diagnostics: dict[str, list[dict[str, Any]]] = {}
            deadline = time.monotonic() + float(timeout_seconds)
            while True:
                if expected_uris.issubset(diagnostics):
                    return dict(sorted(diagnostics.items()))

                reader_failure = getattr(rpc, "_mmm_reader_failure", None)
                if reader_failure is not None:
                    raise java_lsp_module.JDTLanguageServerError(
                        "JDT LS stdout reader failed while collecting diagnostics: "
                        f"{type(reader_failure).__name__}: {reader_failure}"
                    ) from reader_failure

                process = getattr(rpc, "process", None)
                poll = getattr(process, "poll", None)
                if callable(poll):
                    returncode = poll()
                    if returncode is not None:
                        stderr = "\n".join(
                            list(getattr(rpc, "stderr", ())) [-8:]
                        )
                        detail = f"; stderr={stderr}" if stderr else ""
                        raise java_lsp_module.JDTLanguageServerError(
                            "JDT LS exited before publishing complete diagnostics: "
                            f"returncode={returncode}{detail}"
                        )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    message = rpc.messages.get(timeout=min(0.25, remaining))
                except java_lsp_module.queue.Empty:
                    continue
                if java_lsp_module._respond_to_server_request(rpc, message):
                    continue
                if message.get("method") != "textDocument/publishDiagnostics":
                    continue
                params = message.get("params", {})
                if not isinstance(params, dict):
                    continue
                uri = str(params.get("uri", ""))
                if uri not in expected_uris:
                    continue
                values = params.get("diagnostics")
                if not isinstance(values, list):
                    raise java_lsp_module.JDTLanguageServerError(
                        "JDT LS published a malformed diagnostics payload for an "
                        "opened Java file."
                    )
                diagnostics[uri] = java_lsp_module._sorted_diagnostics(
                    item for item in values if isinstance(item, dict)
                )

            missing_count = len(expected_uris.difference(diagnostics))
            raise java_lsp_module.JDTLanguageServerError(
                "JDT LS did not publish diagnostics for every opened Java file "
                "before the validation deadline: "
                f"observed={len(diagnostics)}, expected={len(expected_uris)}, "
                f"missing={missing_count}."
            )

        collect_diagnostics._mmm_jdt_complete_diagnostics = True  # type: ignore[attr-defined]
        collect_diagnostics.__wrapped__ = current_collect  # type: ignore[attr-defined]
        java_lsp_module._collect_diagnostics = collect_diagnostics

    current_init = cls.__init__
    if not getattr(current_init, "_mmm_jdt_process_safety", False):

        @wraps(current_init)
        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            self._mmm_request_lock = threading.RLock()
            self._mmm_write_lock = threading.RLock()
            self._mmm_reader_failure = None
            current_init(self, *args, **kwargs)

        init._mmm_jdt_process_safety = True  # type: ignore[attr-defined]
        cls.__init__ = init

    current_reader = cls._read_stdout
    if not getattr(current_reader, "_mmm_jdt_process_safety", False):

        @wraps(current_reader)
        def read_stdout(self: Any) -> None:
            try:
                current_reader(self)
            except BaseException as exc:
                # Reader exceptions occur on its daemon thread. Persist the failure so
                # the synchronous request path can fail immediately instead of waiting
                # for an unrelated wall-clock timeout.
                self._mmm_reader_failure = exc

        read_stdout._mmm_jdt_process_safety = True  # type: ignore[attr-defined]
        cls._read_stdout = read_stdout

    current_send = cls.send
    if not getattr(current_send, "_mmm_jdt_process_safety", False):

        @wraps(current_send)
        def send(self: Any, payload: dict[str, Any]) -> None:
            with self._mmm_write_lock:
                if self.process.poll() is not None:
                    raise java_lsp_module.JDTLanguageServerError(
                        "JDT LS process exited before the JSON-RPC write: "
                        f"returncode={self.process.returncode}."
                    )
                current_send(self, payload)

        send._mmm_jdt_process_safety = True  # type: ignore[attr-defined]
        cls.send = send

    current_request = cls.request
    if not getattr(current_request, "_mmm_jdt_process_safety", False):

        @wraps(current_request)
        def request(
            self: Any,
            method: str,
            params: dict[str, Any],
            timeout: float,
        ) -> Any:
            if timeout <= 0:
                raise ValueError("JDT LS request timeout must be positive.")
            with self._mmm_request_lock:
                # Reimplement the small request loop so daemon-reader/process failure
                # is checked during every bounded queue wait.
                request_id = self._next_id
                self._next_id += 1
                self.send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )
                deadline = time.monotonic() + float(timeout)
                deferred: list[dict[str, Any]] = []
                try:
                    while True:
                        reader_failure = getattr(self, "_mmm_reader_failure", None)
                        if reader_failure is not None:
                            raise java_lsp_module.JDTLanguageServerError(
                                "JDT LS stdout reader failed: "
                                f"{type(reader_failure).__name__}: {reader_failure}"
                            ) from reader_failure
                        returncode = self.process.poll()
                        if returncode is not None:
                            stderr = "\n".join(list(getattr(self, "stderr", ())) [-8:])
                            detail = f"; stderr={stderr}" if stderr else ""
                            raise java_lsp_module.JDTLanguageServerError(
                                f"JDT LS exited during {method}: returncode={returncode}{detail}"
                            )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(
                                f"JDT LS request timed out: {method}"
                            )
                        try:
                            message = self.messages.get(timeout=min(0.25, remaining))
                        except java_lsp_module.queue.Empty:
                            continue
                        if java_lsp_module._respond_to_server_request(self, message):
                            continue
                        if message.get("id") == request_id:
                            if "error" in message:
                                raise java_lsp_module.JDTLanguageServerError(
                                    str(message["error"])
                                )
                            return message.get("result")
                        deferred.append(message)
                finally:
                    for message in deferred:
                        self.messages.put(message)

        request._mmm_jdt_process_safety = True  # type: ignore[attr-defined]
        request.__wrapped__ = current_request  # type: ignore[attr-defined]
        cls.request = request

    current_close = cls.close
    if not getattr(current_close, "_mmm_jdt_process_safety", False):

        @wraps(current_close)
        def close(self: Any) -> None:
            if self.process.poll() is None:
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
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired as exc:
                        raise java_lsp_module.JDTLanguageServerError(
                            "JDT LS could not be reaped after SIGKILL."
                        ) from exc

            # The process exit closes the pipe ends; bounded joins prevent daemon
            # reader threads from lingering across repeated language-service calls.
            for thread_name in ("_reader", "_error_reader"):
                thread = getattr(self, thread_name, None)
                if isinstance(thread, threading.Thread) and thread is not threading.current_thread():
                    thread.join(timeout=1.0)

        close._mmm_jdt_process_safety = True  # type: ignore[attr-defined]
        cls.close = close


__all__ = ["install"]
