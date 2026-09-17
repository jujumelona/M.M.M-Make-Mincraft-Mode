from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .deadline_executor import collect_completed_with_deadlines
from .source_set_boundary_contract import (
    SourceSetBoundaryError,
    assert_server_safe_source_sets,
)

_DEFAULT_DIAGNOSTIC_PAGE_MAX_FILES = 128
_DEFAULT_DIAGNOSTIC_PAGE_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_DEFAULT_DIAGNOSTIC_QUIET_SECONDS = 2.0
_DEFAULT_PROJECT_JAVA_VERSION = 17
_SEMANTIC_PROBE_NAME = "__MmmJdtReadinessProbe"
_SEMANTIC_PROBE_SOURCE = (
    f"final class {_SEMANTIC_PROBE_NAME} {{\n"
    "    Object objectValue;\n"
    "    String stringValue;\n"
    "}\n"
)
_JAVA_CORE_UNRESOLVED = re.compile(
    r"(?:"
    r"(?:the type\s+)?java\.lang\.(?:Object|String).*cannot be resolved"
    r"|java\.lang\.(?:Object|String).*indirectly referenced"
    r"|(?:^|[^.\w])(?:Object|String)\s+cannot be resolved(?:\s+to\s+a\s+type)?"
    r")",
    re.IGNORECASE,
)


class JDTLanguageServerError(RuntimeError):
    pass


class JDTWorkspaceBootstrapError(JDTLanguageServerError):
    """JDT LS started or configured without a usable Java project bootstrap."""


def _remaining_jdt_deadline(deadline: float, *, operation: str) -> float:
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise JDTLanguageServerError(f"JDT LS {operation} deadline exceeded.")
    return remaining


def _configuration_value(configuration: dict[str, Any], section: str | None) -> Any:
    if not section:
        return configuration
    current: Any = configuration
    for part in section.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _java_executable(java_home: Path) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return java_home / "bin" / f"java{suffix}"


def _parse_java_major(value: str) -> int | None:
    text = value.strip().strip('"').strip("'")
    match = re.search(r"(?:1\.)?(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def _java_major_from_release(java_home: Path) -> int | None:
    release = java_home / "release"
    try:
        text = release.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r'^JAVA_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return _parse_java_major(match.group(1)) if match is not None else None


def _java_major_version(java_home: Path) -> int | None:
    major = _java_major_from_release(java_home)
    if major is not None:
        return major
    executable = _java_executable(java_home)
    if not executable.is_file():
        return None
    try:
        completed = subprocess.run(
            [str(executable), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r'(?:java|openjdk) version "([^"]+)', text)
    if match is None:
        match = re.search(r'^(?:openjdk|java)\s+([^\s]+)', text, re.MULTILINE)
    return _parse_java_major(match.group(1)) if match is not None else None


def _requested_project_java_major() -> int:
    raw = os.environ.get("MMM_JAVA_VERSION", str(_DEFAULT_PROJECT_JAVA_VERSION)).strip()
    major = _parse_java_major(raw)
    if major is None or major <= 0:
        raise JDTWorkspaceBootstrapError(
            f"Invalid MMM_JAVA_VERSION={raw!r}; expected a Java major version."
        )
    return major


def _candidate_java_homes(required_major: int) -> list[Path]:
    candidates: list[Path] = []

    def add(raw: str | Path | None) -> None:
        if raw is None:
            return
        text = str(raw).strip()
        if text:
            candidates.append(Path(text).expanduser())

    for variable in (
        "MMM_PROJECT_JAVA_HOME",
        "MMM_JDT_PROJECT_JAVA_HOME",
        f"JAVA_HOME_{required_major}",
        f"JDK{required_major}_HOME",
        f"JDK_{required_major}_HOME",
        "JAVA_HOME",
    ):
        add(os.environ.get(variable))

    discovered = shutil.which("java")
    if discovered:
        add(Path(discovered).resolve().parent.parent)

    if os.name == "nt":
        roots = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ]
        patterns = (
            "Java/*",
            "Eclipse Adoptium/*",
            "Microsoft/jdk-*",
            "Programs/Eclipse Adoptium/*",
        )
        for root in roots:
            if not root:
                continue
            base = Path(root)
            for pattern in patterns:
                candidates.extend(base.glob(pattern))
    elif os.uname().sysname == "Darwin":
        candidates.extend(Path("/Library/Java/JavaVirtualMachines").glob("*/Contents/Home"))
        candidates.extend(Path.home().glob("Library/Java/JavaVirtualMachines/*/Contents/Home"))
    else:
        candidates.extend(Path("/usr/lib/jvm").glob("*"))
        candidates.extend(Path("/opt").glob("jdk*"))
        candidates.extend(Path.home().glob(".jdks/*"))
        candidates.extend(Path.home().glob(".sdkman/candidates/java/*"))

    return candidates


def _java_major_versions(java_homes: list[Path]) -> list[int | None]:
    if not java_homes:
        return []
    workers = max(1, min(len(java_homes), os.cpu_count() or 1))
    if workers == 1:
        return [_java_major_version(java_homes[0])]
    completed = collect_completed_with_deadlines(
        java_homes,
        _java_major_version,
        max_workers=workers,
        stage="jdt-jdk-probe",
        sort_key=lambda home: home.as_posix(),
    )
    versions_by_home = {home: major for home, major in completed}
    return [versions_by_home[home] for home in java_homes]


def _resolve_project_java_home(required_major: int | None=None) -> Path:
    return _mmm__resolve_project_java_home_impl(required_major)

def _mmm__resolve_project_java_home_impl(required_major: int | None = None) -> Path:
    required = required_major if required_major is not None else _requested_project_java_major()
    seen: set[Path] = set()
    homes: list[Path] = []
    for candidate in _candidate_java_homes(required):
        try:
            home = candidate.resolve(strict=True)
        except OSError:
            continue
        if home in seen or not home.is_dir():
            continue
        seen.add(home)
        homes.append(home)

    majors = _java_major_versions(homes)
    observed: list[str] = []
    for home, major in zip(homes, majors, strict=True):
        if major is None:
            continue
        observed.append(f"{home}=>{major}")
        if major == required:
            return home
    detail = ", ".join(observed) if observed else "no usable Java homes discovered"
    try:
        from .jdtls_bootstrap import JDTLSBootstrapError, ensure_project_jdk

        provisioned = ensure_project_jdk(required)
    except (JDTLSBootstrapError, OSError, ValueError) as exc:
        raise JDTWorkspaceBootstrapError(
            "JDT workspace bootstrap failure: no project JDK matching "
            f"MMM_JAVA_VERSION={required} was found locally ({detail}); "
            f"lazy provisioning failed: {type(exc).__name__}: {exc}"
        ) from exc
    if provisioned is not None:
        return provisioned.resolve()
    raise JDTWorkspaceBootstrapError(
        "JDT workspace bootstrap failure: no project JDK matching "
        f"MMM_JAVA_VERSION={required} was found ({detail})."
    )


def _java_runtime_name(major: int) -> str:
    return "JavaSE-1.8" if major == 8 else f"JavaSE-{major}"


def _project_java_runtime(project_java_home: Path | None = None) -> dict[str, Any]:
    required = _requested_project_java_major()
    home = project_java_home or _resolve_project_java_home(required)
    actual = _java_major_version(home)
    if actual != required:
        raise JDTWorkspaceBootstrapError(
            "JDT workspace bootstrap failure: resolved project JDK version mismatch: "
            f"required={required}, actual={actual}, home={home}."
        )
    return {
        "name": _java_runtime_name(required),
        "path": str(home),
        "default": True,
    }


def _jdt_configuration(project_java_home: Path | None = None) -> dict[str, Any]:
    runtime = _project_java_runtime(project_java_home)
    return {
        "java": {
            "autobuild": {"enabled": True},
            "configuration": {
                "updateBuildConfiguration": "automatic",
                "runtimes": [runtime],
            },
            "import": {
                "gradle": {"enabled": True, "java": {"home": runtime["path"]}}
            },
        }
    }


def _jdtls_environment() -> dict[str, str]:
    env = dict(os.environ)
    launcher_home = os.environ.get("MMM_JDTLS_JAVA_HOME", "").strip()
    if launcher_home:
        home = Path(launcher_home).expanduser().resolve()
        executable = _java_executable(home)
        if not executable.is_file():
            raise JDTLanguageServerError(
                f"MMM_JDTLS_JAVA_HOME does not contain a Java launcher: {home}"
            )
        env["JAVA_HOME"] = str(home)
        env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


class _JsonRpcProcess:
    def __init__(
        self,
        command: list[str],
        cwd: Path,
        *,
        configuration: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=environment,
        )
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr: deque[str] = deque(maxlen=30)
        self.workspace_folders = [{"uri": cwd.resolve().as_uri(), "name": cwd.name}]
        self.configuration = configuration or {}
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
                    message = self.messages.get(timeout=min(0.25, max(0.001, deadline - time.monotonic())))
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
        try:
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
        except BaseException as exc:
            self._mmm_reader_failure = exc

    def _read_stderr(self) -> None:
        from .agent_tool_runtime import _redact_text
        from .root_cause_trace import emit_root_cause

        stream = self.process.stderr
        if stream is None:
            return
        private_key_block = False
        for raw in iter(stream.readline, b""):
            decoded = raw.decode("utf-8", errors="replace").rstrip()
            if private_key_block:
                if re.search(r"-----END [^-]*PRIVATE KEY-----", decoded):
                    private_key_block = False
                continue
            if re.search(r"-----BEGIN [^-]*PRIVATE KEY-----", decoded):
                private_key_block = not bool(re.search(r"-----END [^-]*PRIVATE KEY-----", decoded))
                line = "[REDACTED_PRIVATE_KEY]"
            else:
                line = _redact_text(decoded)
            self.stderr.append(line)
            emit_root_cause(
                "jdt_stderr",
                stage="jdt",
                operation="stderr_reader",
                gate="jdt_process_stderr",
                result="INFO",
                details={"pid": self.process.pid, "line": line},
            )


def _respond_to_server_request(rpc: _JsonRpcProcess, message: dict[str, Any]) -> bool:
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
        result = []
        if isinstance(items, list):
            for item in items:
                section = item.get("section") if isinstance(item, dict) else None
                result.append(_configuration_value(rpc.configuration, section if isinstance(section, str) else None))
    elif method == "workspace/workspaceFolders":
        result = list(rpc.workspace_folders)
    elif method == "workspace/applyEdit":
        result = {"applied": False}
    else:
        rpc.send({
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {"code": -32601, "message": f"Unsupported server request: {method}"},
        })
        return True

    rpc.send({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
    return True


def _initialize_rpc_session(
    rpc: _JsonRpcProcess,
    root: Path,
    configuration: dict[str, Any],
    *,
    timeout_seconds: int,
    quiet_seconds: float,
    deadline: float,
) -> None:
    try:
        rpc.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root.as_uri(),
                "capabilities": {
                    "textDocument": {"publishDiagnostics": {"relatedInformation": True}},
                    "workspace": {
                        "configuration": True,
                        "workspaceFolders": True,
                        "symbol": {},
                    },
                },
                "initializationOptions": {
                    "settings": configuration,
                    "extendedClientCapabilities": {"progressReportProvider": True},
                },
                "workspaceFolders": list(rpc.workspace_folders),
            },
            timeout=min(_remaining_jdt_deadline(deadline, operation="initialize"), 45.0),
        )
        rpc.notify("initialized", {})
        rpc.notify("workspace/didChangeConfiguration", {"settings": configuration})
        _await_java_core_ready(
            rpc,
            root,
            timeout_seconds=timeout_seconds,
            quiet_seconds=quiet_seconds,
            deadline=deadline,
        )
    except BaseException:
        rpc.close()
        raise


class JavaLanguageService:
    """Bounded Eclipse JDT LS client with fail-closed project-JDK readiness."""

    def __init__(
        self,
        command: str | None = None,
        *,
        diagnostic_page_max_files: int = _DEFAULT_DIAGNOSTIC_PAGE_MAX_FILES,
        diagnostic_page_max_source_bytes: int = _DEFAULT_DIAGNOSTIC_PAGE_MAX_SOURCE_BYTES,
        diagnostic_quiet_seconds: float = _DEFAULT_DIAGNOSTIC_QUIET_SECONDS,
    ) -> None:
        raw = command or os.environ.get("MMM_JDTLS_CMD", "").strip()
        self.command = shlex.split(raw) if raw else ["jdtls"]
        if not self.command:
            raise JDTLanguageServerError("Empty JDT LS command.")
        if diagnostic_page_max_files <= 0:
            raise ValueError("diagnostic_page_max_files must be positive.")
        if diagnostic_page_max_source_bytes <= 0:
            raise ValueError("diagnostic_page_max_source_bytes must be positive.")
        if diagnostic_quiet_seconds < 0:
            raise ValueError("diagnostic_quiet_seconds cannot be negative.")
        self.diagnostic_page_max_files = diagnostic_page_max_files
        self.diagnostic_page_max_source_bytes = diagnostic_page_max_source_bytes
        self.diagnostic_quiet_seconds = diagnostic_quiet_seconds
        self._session_lock = threading.RLock()
        self._rpc: _JsonRpcProcess | None = None
        self._project_root: Path | None = None
        self._project_java_home: Path | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and self._rpc is not None and self._rpc_alive(self._rpc)

    @staticmethod
    def _rpc_alive(rpc: _JsonRpcProcess) -> bool:
        process = getattr(rpc, "process", None)
        poll = getattr(process, "poll", None)
        return not callable(poll) or poll() is None

    def _close_rpc_locked(self) -> None:
        rpc = self._rpc
        self._rpc = None
        self._project_root = None
        self._project_java_home = None
        self._ready = False
        if rpc is not None:
            rpc.close()

    def _ensure_rpc_locked(
        self,
        root: Path,
        *,
        timeout_seconds: int,
        deadline: float | None = None,
    ) -> _JsonRpcProcess:
        deadline = deadline if deadline is not None else time.monotonic() + float(timeout_seconds)
        _remaining_jdt_deadline(deadline, operation="session initialization")
        rpc = self._rpc
        if rpc is not None and self._project_root == root and self.ready:
            return rpc
        if rpc is not None:
            self._close_rpc_locked()

        project_java_home = _resolve_project_java_home()
        configuration = _jdt_configuration(project_java_home)
        rpc = _JsonRpcProcess(
            self.command,
            root,
            configuration=configuration,
            environment=_jdtls_environment(),
        )
        _initialize_rpc_session(
            rpc,
            root,
            configuration,
            timeout_seconds=timeout_seconds,
            quiet_seconds=min(self.diagnostic_quiet_seconds, 0.25),
            deadline=deadline,
        )
        self._rpc = rpc
        self._project_root = root
        self._project_java_home = project_java_home
        self._ready = True
        return rpc

    def close(self) -> None:
        with self._session_lock:
            self._close_rpc_locked()

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
        deadline = time.monotonic() + float(timeout_seconds)
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
            _remaining_jdt_deadline(deadline, operation="diagnostics")
            rpc = self._ensure_rpc_locked(
                root,
                timeout_seconds=timeout_seconds,
                deadline=deadline,
            )
            diagnostics: dict[str, list[dict[str, Any]]] = {}
            page_receipts: list[dict[str, Any]] = []
            total_source_bytes = 0
            for page_index, page in enumerate(pages):
                _remaining_jdt_deadline(deadline, operation="diagnostics")
                sources, source_bytes = _read_source_page(
                    page,
                    max_source_bytes=self.diagnostic_page_max_source_bytes,
                )
                expected_uris = {path.as_uri() for path, _text in sources}
                for source_path, source_text in sources:
                    rpc.notify("textDocument/didOpen", {"textDocument": {
                        "uri": source_path.as_uri(),
                        "languageId": "java",
                        "version": 1,
                        "text": source_text,
                    }})
                try:
                    page_diagnostics = _collect_diagnostics(
                        rpc,
                        expected_uris=expected_uris,
                        timeout_seconds=_remaining_jdt_deadline(
                            deadline, operation="diagnostics"
                        ),
                        quiet_seconds=self.diagnostic_quiet_seconds,
                        deadline=deadline,
                    )
                    _raise_on_java_core_bootstrap_failure(page_diagnostics)
                finally:
                    for source_path, _source_text in sources:
                        rpc.notify(
                            "textDocument/didClose",
                            {"textDocument": {"uri": source_path.as_uri()}},
                        )
                diagnostics.update(page_diagnostics)
                page_errors, page_warnings = _diagnostic_counts(page_diagnostics)
                relative_paths = [source_path.relative_to(root).as_posix() for source_path, _ in sources]
                page_receipts.append({
                    "page_index": page_index,
                    "file_count": len(sources),
                    "source_bytes": source_bytes,
                    "first_file": relative_paths[0],
                    "last_file": relative_paths[-1],
                    "diagnostic_uri_count": len(page_diagnostics),
                    "error_count": page_errors,
                    "warning_count": page_warnings,
                })
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

    def workspace_symbols(
        self,
        project_root: str | Path,
        query: str,
        *,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        deadline = time.monotonic() + float(timeout_seconds)
        with self._session_lock:
            _remaining_jdt_deadline(deadline, operation="workspace symbols")
            rpc = self._ensure_rpc_locked(
                root,
                timeout_seconds=timeout_seconds,
                deadline=deadline,
            )
            result = rpc.request(
                "workspace/symbol",
                {"query": query},
                timeout=_remaining_jdt_deadline(deadline, operation="workspace symbols"),
            )
            return {
                "schema_version": "mmm/java-symbols-v1",
                "query": query,
                "symbols": result or [],
            }


JavaLanguageService.diagnostics.__mmm_source_set_boundary__ = True


def _validated_java_file(root: Path, candidate: Path) -> Path:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Java file escaped the project root.") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("Java file path is not a canonical project-relative path.")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Java file path traversed a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(candidate) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Java file escaped the project root.") from exc
    if resolved.suffix != ".java" or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _java_files(root: Path, relative_files: Iterable[str] | None) -> list[Path]:
    if relative_files is None:
        candidates = (
            _validated_java_file(root, path)
            for path in root.rglob("*.java")
            if path.is_file()
        )
    else:
        requested: list[Path] = []
        for relative in relative_files:
            raw = Path(relative)
            if raw.is_absolute():
                raise ValueError("Java file path must be project-relative.")
            requested.append(_validated_java_file(root, root / raw))
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
            len(current) >= max_files or current_bytes + source_bytes > max_source_bytes
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
                "Java sources changed while preparing a JDT LS page and now exceed its source-byte limit."
            )
        sources.append((path, raw.decode("utf-8", errors="replace")))
    return sources, total_bytes


def _java_source_root(root: Path) -> Path:
    for relative in ("src/main/java", "src/client/java", "src/test/java"):
        candidate = root / relative
        if candidate.is_dir():
            return candidate.resolve()
    files = _java_files(root, None)
    if files:
        return files[0].parent
    raise JDTWorkspaceBootstrapError(
        "JDT workspace bootstrap failure: no Java source root exists for the semantic readiness probe."
    )


def _java_core_bootstrap_messages(
    diagnostics: dict[str, list[dict[str, Any]]],
) -> list[str]:
    messages: list[str] = []
    for values in diagnostics.values():
        for item in values:
            message = str(item.get("message", ""))
            if _JAVA_CORE_UNRESOLVED.search(message):
                messages.append(message)
    return sorted(set(messages))


def _raise_on_java_core_bootstrap_failure(
    diagnostics: dict[str, list[dict[str, Any]]],
) -> None:
    messages = _java_core_bootstrap_messages(diagnostics)
    if not messages:
        return
    detail = "; ".join(messages[-3:])
    raise JDTWorkspaceBootstrapError(
        "JDT workspace bootstrap failure: java.lang.Object/java.lang.String cannot be resolved. "
        f"Diagnostics: {detail}"
    )


def _await_java_core_ready(
    rpc: _JsonRpcProcess,
    root: Path,
    *,
    timeout_seconds: float,
    quiet_seconds: float,
    deadline: float | None = None,
) -> None:
    """Wait for compiler diagnostics to prove core Java types resolve.

    Hover is intentionally not used as a readiness signal. It is an optional language
    feature request and can block independently of compiler diagnostics while JDT is
    importing a Gradle workspace.
    """

    source_root = _java_source_root(root)
    probe_path = source_root / f"{_SEMANTIC_PROBE_NAME}.java"
    uri = probe_path.resolve(strict=False).as_uri()
    deadline = deadline if deadline is not None else time.monotonic() + float(timeout_seconds)
    version = 1
    last_reason = "semantic diagnostics have not completed"

    while time.monotonic() < deadline:
        rpc.notify("textDocument/didOpen", {"textDocument": {
            "uri": uri,
            "languageId": "java",
            "version": version,
            "text": _SEMANTIC_PROBE_SOURCE,
        }})
        try:
            remaining = _remaining_jdt_deadline(deadline, operation="semantic readiness")
            diagnostics = _collect_diagnostics(
                rpc,
                expected_uris={uri},
                timeout_seconds=remaining,
                quiet_seconds=quiet_seconds,
                deadline=deadline,
            )
            core_messages = _java_core_bootstrap_messages(diagnostics)
            errors = [
                item
                for values in diagnostics.values()
                for item in values
                if int(item.get("severity", 1)) == 1
            ]
            if not core_messages and not errors:
                return
            if core_messages:
                last_reason = "; ".join(core_messages[-3:])
            else:
                last_reason = "; ".join(
                    str(item.get("message", "")) for item in errors[-3:]
                )
        except (JDTLanguageServerError, TimeoutError) as exc:
            last_reason = f"{type(exc).__name__}: {exc}"
        finally:
            rpc.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

        version += 1
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.25, remaining))

    raise JDTWorkspaceBootstrapError(
        "JDT workspace bootstrap failure: initialize/status notifications were insufficient; "
        "project-source diagnostics did not prove java.lang.Object/java.lang.String readiness "
        f"before validation. Last probe state: {last_reason}"
    )


def _raise_diagnostic_transport_failure(rpc: _JsonRpcProcess) -> None:
    reader_failure = getattr(rpc, "_mmm_reader_failure", None)
    if reader_failure is not None:
        raise JDTLanguageServerError(
            "JDT LS stdout reader failed while collecting diagnostics: "
            f"{type(reader_failure).__name__}: {reader_failure}"
        ) from reader_failure
    process = getattr(rpc, "process", None)
    poll = getattr(process, "poll", None)
    if not callable(poll):
        return
    returncode = poll()
    if returncode is None:
        return
    stderr = "\n".join(list(getattr(rpc, "stderr", ()))[-8:])
    detail = f"; stderr={stderr}" if stderr else ""
    raise JDTLanguageServerError(
        "JDT LS exited before publishing complete diagnostics: "
        f"returncode={returncode}{detail}"
    )


def _diagnostic_wait_seconds(
    *,
    deadline: float,
    now: float,
    complete: bool,
    settled_since: float | None,
    quiet_seconds: float,
) -> float | None:
    remaining = deadline - now
    if remaining <= 0:
        return None
    wait_seconds = min(0.25, remaining)
    if complete and settled_since is not None:
        settle_remaining = quiet_seconds - (now - settled_since)
        return min(wait_seconds, max(0.001, settle_remaining))
    return wait_seconds


def _published_diagnostics(
    message: dict[str, Any],
    expected_uris: set[str],
) -> tuple[str, list[dict[str, Any]]] | None:
    if message.get("method") != "textDocument/publishDiagnostics":
        return None
    params = message.get("params", {})
    if not isinstance(params, dict):
        return None
    uri = str(params.get("uri", ""))
    if uri not in expected_uris:
        return None
    values = params.get("diagnostics")
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise JDTLanguageServerError(
            "JDT LS published a malformed diagnostics payload for an opened Java file."
        )
    return uri, _sorted_diagnostics(values)


def _raise_diagnostic_deadline(
    expected_uris: set[str],
    diagnostics: dict[str, list[dict[str, Any]]],
) -> None:
    missing_count = len(expected_uris.difference(diagnostics))
    if missing_count:
        raise JDTLanguageServerError(
            "JDT LS did not publish diagnostics for every opened Java file before the validation deadline: "
            f"observed={len(diagnostics)}, expected={len(expected_uris)}, missing={missing_count}."
        )
    raise JDTLanguageServerError(
        "JDT LS diagnostics did not become quiescent before the validation deadline "
        f"after all {len(expected_uris)} opened Java files were observed."
    )


def _collect_diagnostics(
    rpc: _JsonRpcProcess,
    *,
    expected_uris: set[str],
    timeout_seconds: float,
    quiet_seconds: float,
    deadline: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if timeout_seconds <= 0:
        raise ValueError("JDT diagnostics timeout must be positive.")
    if quiet_seconds < 0:
        raise ValueError("JDT diagnostics quiet period cannot be negative.")
    if not expected_uris:
        return {}

    diagnostics: dict[str, list[dict[str, Any]]] = {}
    deadline = deadline if deadline is not None else time.monotonic() + float(timeout_seconds)
    settled_since: float | None = None
    while True:
        now = time.monotonic()
        complete = expected_uris.issubset(diagnostics)
        if complete and settled_since is not None and now - settled_since >= quiet_seconds:
            return dict(sorted(diagnostics.items()))
        _raise_diagnostic_transport_failure(rpc)
        wait_seconds = _diagnostic_wait_seconds(
            deadline=deadline,
            now=now,
            complete=complete,
            settled_since=settled_since,
            quiet_seconds=quiet_seconds,
        )
        if wait_seconds is None:
            break
        try:
            message = rpc.messages.get(timeout=wait_seconds)
        except queue.Empty:
            continue
        if _respond_to_server_request(rpc, message):
            continue
        published = _published_diagnostics(message, expected_uris)
        if published is None:
            continue
        uri, values = published
        diagnostics[uri] = values
        settled_since = time.monotonic()

    _raise_diagnostic_deadline(expected_uris, diagnostics)
    raise AssertionError("unreachable")


def _sorted_diagnostics(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
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
