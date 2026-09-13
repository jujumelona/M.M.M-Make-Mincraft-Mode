from __future__ import annotations

"""Host-owned incremental Java generation verification.

The host owns verifier scope and deadlines. A persistent JDT session establishes the
baseline once, then only changed/new Java sources are diagnosed. Java deletions and
project-model changes invalidate the session and force a fresh full scan. JDT failure
is a verifier infrastructure failure; there is deliberately no alternate Gradle
verification backend or run-local circuit breaker.
"""

import hashlib
import json
import os
import queue
import time
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

from .root_cause_trace import emit_root_cause

_MARKER = "_mmm_host_owned_generation_verifier"
_JDT_COLLECTOR_MARKER = "_mmm_progress_aware_jdt_collector"
_VERIFIER_NAME = "java_diagnostics"
_JDT_SERVICE_ATTR = "_mmm_generation_java_service"
_JAVA_SNAPSHOT_ATTR = "_mmm_generation_java_snapshot"
_MODEL_SNAPSHOT_ATTR = "_mmm_generation_model_snapshot"
_MODEL_FILES = (
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.properties",
    "gradle/libs.versions.toml",
    "pom.xml",
    ".classpath",
    ".project",
)


def _bounded_int_env(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def host_jdt_idle_timeout_seconds() -> int:
    """Maximum silence between meaningful JDT messages during generation verification."""

    return _bounded_int_env(
        "MMM_JDT_DIAGNOSTIC_IDLE_TIMEOUT_SECONDS",
        default=90,
        minimum=15,
        maximum=300,
    )


def host_jdt_hard_timeout_seconds(idle_timeout_seconds: float) -> int:
    """Absolute safety cap; progress may extend work only up to this host-owned bound."""

    default = max(600, int(float(idle_timeout_seconds) * 4.0))
    value = _bounded_int_env(
        "MMM_JDT_DIAGNOSTIC_HARD_TIMEOUT_SECONDS",
        default=default,
        minimum=60,
        maximum=1800,
    )
    return max(int(float(idle_timeout_seconds)), value)


def _forced_tool_name(tool_choice: Any) -> str:
    if not isinstance(tool_choice, Mapping):
        return ""
    function = tool_choice.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name") or "").strip()


def synthesized_verifier_turn(messages: list[dict[str, Any]]) -> Any:
    """Create the mechanical verifier call without spending a coder inference turn."""

    from .model_adapters import GenerationResponse, ToolCall

    arguments = {"timeout_seconds": host_jdt_idle_timeout_seconds()}
    raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    identity_material = f"{len(messages)}:{raw_arguments}".encode("utf-8")
    call_id = "host_verify_" + hashlib.sha256(identity_material).hexdigest()[:16]
    return GenerationResponse(
        content="",
        tool_calls=(
            ToolCall(
                id=call_id,
                name=_VERIFIER_NAME,
                arguments=arguments,
                raw_arguments=raw_arguments,
            ),
        ),
    )


def _normalize_relative_files(raw_files: Any) -> list[str] | None:
    if raw_files is None:
        return None
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Verifier relative_files must be a non-empty list when supplied")
    normalized: list[str] = []
    for raw_file in raw_files:
        relative = str(raw_file or "").replace("\\", "/").strip()
        while relative.startswith("./"):
            relative = relative[2:]
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not relative.endswith(".java")
        ):
            raise ValueError(f"Invalid task diagnostic path: {relative!r}")
        normalized.append(relative)
    return list(dict.fromkeys(normalized))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _java_snapshot(project_root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(project_root.rglob("*.java"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(project_root).as_posix()
        snapshot[relative] = _sha256_file(path)
    return snapshot


def _model_snapshot(project_root: Path) -> tuple[tuple[str, str], ...]:
    candidates = [project_root / relative for relative in _MODEL_FILES]
    settings = project_root / ".settings"
    if settings.is_dir() and not settings.is_symlink():
        candidates.extend(sorted(settings.glob("*.prefs"), key=lambda item: item.as_posix()))
    snapshot: list[tuple[str, str]] = []
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            snapshot.append((path.relative_to(project_root).as_posix(), _sha256_file(path)))
    return tuple(snapshot)


def _close_generation_jdt(runtime: Any) -> None:
    service = getattr(runtime, _JDT_SERVICE_ATTR, None)
    try:
        if service is not None:
            close = getattr(service, "close", None)
            if callable(close):
                close()
    finally:
        try:
            delattr(runtime, _JDT_SERVICE_ATTR)
        except AttributeError:
            pass


def _unchanged_receipt(project_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "mmm/java-diagnostics-v2",
        "project_root": str(project_root),
        "files_opened": 0,
        "total_source_bytes": 0,
        "page_count": 0,
        "pages": [],
        "error_count": 0,
        "warning_count": 0,
        "diagnostics": {},
        "server_stderr_tail": [],
        "verification_backend": "jdt_host",
        "verification_scope": "unchanged",
        "verified_files": [],
        "skipped": True,
    }


def run_generation_verifier(
    runtime: Any,
    arguments: Mapping[str, Any] | None,
    *,
    runtime_module: Any,
    java_service_factory: Any | None = None,
) -> dict[str, Any]:
    """Run fail-closed incremental JDT verification under host-owned scope."""

    from .java_lsp import JDTLanguageServerError
    from .java_lsp_trace import TracedJavaLanguageService

    payload = dict(arguments or {})
    raw_payload = dict(payload)
    project_root, _project_argument = runtime_module._discover_model_project_root(
        runtime.workspace_root
    )
    try:
        requested_files = _normalize_relative_files(payload.get("relative_files"))
    except ValueError as exc:
        raise runtime_module.AgentToolRuntimeError(str(exc)) from exc

    full_scan = bool(payload.get("full_scan", False))
    idle_timeout = host_jdt_idle_timeout_seconds()
    current_java = _java_snapshot(project_root)
    current_model = _model_snapshot(project_root)
    previous_java = getattr(runtime, _JAVA_SNAPSHOT_ATTR, None)
    previous_model = getattr(runtime, _MODEL_SNAPSHOT_ATTR, None)

    first_scan = previous_java is None
    model_changed = previous_model is not None and previous_model != current_model
    deleted_files = (
        sorted(set(previous_java).difference(current_java))
        if isinstance(previous_java, dict)
        else []
    )
    changed_files = (
        sorted(
            path
            for path, digest in current_java.items()
            if not isinstance(previous_java, dict) or previous_java.get(path) != digest
        )
        if not first_scan
        else sorted(current_java)
    )

    reset_session = bool(model_changed or deleted_files)
    if reset_session:
        _close_generation_jdt(runtime)

    force_full = bool(full_scan or first_scan or reset_session)
    if force_full:
        relative_files: list[str] | None = None
        scope = "full"
    else:
        targets = set(changed_files)
        if requested_files is not None:
            targets.update(path for path in requested_files if path in current_java)
        relative_files = sorted(targets)
        scope = "incremental"

    normalized: dict[str, Any] = {
        "project_root": str(project_root),
        "timeout_seconds": idle_timeout,
        "verification_scope": scope if relative_files or force_full else "unchanged",
    }
    if relative_files is not None:
        normalized["relative_files"] = relative_files

    emit_root_cause(
        "generation_verifier_host_dispatch",
        stage="generation",
        operation=_VERIFIER_NAME,
        gate="host_verifier_authority",
        result="START",
        reason="host selected full/incremental Java verification scope from filesystem state",
        details={
            "raw_arguments": raw_payload,
            "normalized_arguments": normalized,
            "workspace_root": runtime.workspace_root,
            "model_changed": model_changed,
            "deleted_files": deleted_files,
            "changed_files": changed_files,
        },
    )

    if relative_files == []:
        result = _unchanged_receipt(project_root)
        setattr(runtime, _JAVA_SNAPSHOT_ATTR, current_java)
        setattr(runtime, _MODEL_SNAPSHOT_ATTR, current_model)
        emit_root_cause(
            "generation_verifier_unchanged_skip",
            stage="generation",
            operation=_VERIFIER_NAME,
            gate="incremental_scope",
            result="SKIP",
            reason="no Java source or project-model changes since the last successful verification",
        )
        return runtime_module._bounded_result(result)

    try:
        service = getattr(runtime, _JDT_SERVICE_ATTR, None)
        if service is None:
            factory = java_service_factory or TracedJavaLanguageService
            service = factory()
            setattr(runtime, _JDT_SERVICE_ATTR, service)
        result = dict(
            service.diagnostics(
                project_root,
                relative_files=relative_files,
                timeout_seconds=idle_timeout,
            )
        )
    except (OSError, TimeoutError, JDTLanguageServerError) as jdt_exc:
        _close_generation_jdt(runtime)
        emit_root_cause(
            "generation_verifier_jdt_unavailable",
            stage="generation",
            operation=_VERIFIER_NAME,
            gate="jdt_host_verifier",
            result="FAIL",
            reason=f"{type(jdt_exc).__name__}: {jdt_exc}",
            exc=jdt_exc,
        )
        raise runtime_module.AgentToolRuntimeError(
            "Generation verification failed because JDT is unavailable: "
            f"{type(jdt_exc).__name__}: {jdt_exc}"
        ) from jdt_exc

    result.setdefault("verification_backend", "jdt_host")
    result["verification_scope"] = scope
    result["verified_files"] = sorted(current_java) if force_full else list(relative_files or ())
    result["skipped"] = False
    setattr(runtime, _JAVA_SNAPSHOT_ATTR, current_java)
    setattr(runtime, _MODEL_SNAPSHOT_ATTR, current_model)
    emit_root_cause(
        "generation_verifier_jdt_result",
        stage="generation",
        operation=_VERIFIER_NAME,
        gate="jdt_host_verifier",
        result="PASS",
        details={"result": result},
    )
    return runtime_module._bounded_result(result)


def _collect_diagnostics_progress_aware(
    rpc: Any,
    *,
    expected_uris: set[str],
    timeout_seconds: float,
    quiet_seconds: float,
    page_index: int,
) -> dict[str, list[dict[str, Any]]]:
    """Wait on JDT silence, not total import duration, with an independent hard cap."""

    from . import java_lsp_trace as trace_module
    from .java_lsp import JDTLanguageServerError, _respond_to_server_request, _sorted_diagnostics

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
    last_progress = started
    hard_timeout = float(host_jdt_hard_timeout_seconds(timeout_seconds))
    hard_deadline = started + hard_timeout
    settled_since: float | None = None
    timeout_kind = "idle"

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
                    "idle_ms": round((now - last_progress) * 1000.0, 3),
                    "hard_timeout_seconds": hard_timeout,
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

        returncode = rpc.process.poll()
        if returncode is not None:
            stderr = "\n".join(list(rpc.stderr)[-8:])
            raise JDTLanguageServerError(
                "JDT LS exited before publishing complete diagnostics: "
                f"returncode={returncode}; stderr={stderr or '<empty>'}"
            )

        idle_remaining = float(timeout_seconds) - (now - last_progress)
        hard_remaining = hard_deadline - now
        if hard_remaining <= 0:
            timeout_kind = "hard"
            break
        if idle_remaining <= 0:
            timeout_kind = "idle"
            break
        wait_seconds = min(0.25, idle_remaining, hard_remaining)
        if complete and settled_since is not None:
            wait_seconds = min(wait_seconds, max(0.001, quiet_seconds - (now - settled_since)))
        try:
            message = rpc.messages.get(timeout=max(0.001, wait_seconds))
        except queue.Empty:
            continue

        now = time.monotonic()
        if _respond_to_server_request(rpc, message):
            last_progress = now
            ignored_methods[str(message.get("method") or "<server-request>")] += 1
            continue

        method = str(message.get("method") or "<response>")
        if method != "textDocument/publishDiagnostics":
            trace_module._record_server_progress(rpc, message)
            ignored_methods[method] += 1
            last_progress = now
            continue

        params = message.get("params")
        if not isinstance(params, dict):
            malformed_messages += 1
            raise JDTLanguageServerError("JDT LS publishDiagnostics params were not an object.")
        uri = str(params.get("uri") or "")
        if uri not in expected_uris:
            unexpected_uris.add(uri or "<missing-uri>")
            last_progress = now
            continue
        values = params.get("diagnostics")
        if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
            malformed_messages += 1
            raise JDTLanguageServerError(
                f"JDT LS published malformed diagnostics for expected URI {uri!r}."
            )
        diagnostics[uri] = _sorted_diagnostics(values)
        last_progress = now
        settled_since = now

    missing_uris = sorted(expected_uris.difference(diagnostics))
    state = {
        "page_index": page_index,
        "timeout_kind": timeout_kind,
        "idle_timeout_seconds": float(timeout_seconds),
        "hard_timeout_seconds": hard_timeout,
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
        "idle_ms": round((time.monotonic() - last_progress) * 1000.0, 3),
        "server_progress_tail": list(getattr(rpc, "server_progress_tail", ())),
    }
    event = "jdt_publish_timeout" if missing_uris else "jdt_quiet_timeout"
    emit_root_cause(
        event,
        stage="jdt",
        operation="diagnostics",
        gate="publishDiagnostics" if missing_uris else "diagnostic_quiescence",
        result="FAIL",
        reason=f"JDT diagnostics {timeout_kind} timeout",
        details=state,
    )
    if missing_uris:
        raise JDTLanguageServerError(
            "JDT LS stopped making diagnostic progress before every opened Java file "
            f"was observed ({timeout_kind} timeout): observed={len(diagnostics)}, "
            f"expected={len(expected_uris)}, missing={len(missing_uris)}; state={state}"
        )
    raise JDTLanguageServerError(
        "JDT LS published every opened Java file but did not become quiescent before "
        f"the {timeout_kind} timeout; state={state}"
    )


def install(
    *,
    agent_tool_runtime_module: Any,
    progress_loop_module: Any,
    java_lsp_trace_module: Any,
) -> None:
    """Install the final host-owned verifier path after runtime composition."""

    runtime_cls = agent_tool_runtime_module.AgentToolRuntime
    current_call = runtime_cls._call
    if not getattr(current_call, _MARKER, False):

        @wraps(current_call)
        def call_with_host_generation_verifier(
            self: Any,
            stage: str,
            name: str,
            arguments: Mapping[str, Any] | None,
            *,
            external_server_ids: frozenset[str] | None,
        ) -> dict[str, Any]:
            selected = self._stage(stage)
            if selected == "generation" and str(name).strip() == _VERIFIER_NAME and external_server_ids is None:
                return run_generation_verifier(
                    self,
                    arguments,
                    runtime_module=agent_tool_runtime_module,
                )
            return current_call(
                self,
                stage,
                name,
                arguments,
                external_server_ids=external_server_ids,
            )

        setattr(call_with_host_generation_verifier, _MARKER, True)
        call_with_host_generation_verifier.__wrapped__ = current_call
        runtime_cls._call = call_with_host_generation_verifier

    current_turn = progress_loop_module._generate_turn_with_context_recovery
    if not getattr(current_turn, _MARKER, False):

        @wraps(current_turn)
        def generate_turn_with_host_verifier(
            router: Any,
            *,
            config: Any,
            adapter: Any,
            request: Any,
            messages: list[dict[str, Any]],
            media_paths: tuple[Any, ...],
            tool_choice: Any,
            parallel_tool_calls: bool,
        ) -> Any:
            if _forced_tool_name(tool_choice) == _VERIFIER_NAME:
                turn = synthesized_verifier_turn(messages)
                emit_root_cause(
                    "generation_verifier_model_turn_elided",
                    stage="generation",
                    operation=_VERIFIER_NAME,
                    gate="host_verifier_authority",
                    result="PASS",
                    reason="forced verifier selection is mechanical and does not require coder inference",
                )
                return turn
            return current_turn(
                router,
                config=config,
                adapter=adapter,
                request=request,
                messages=messages,
                media_paths=media_paths,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )

        setattr(generate_turn_with_host_verifier, _MARKER, True)
        generate_turn_with_host_verifier.__wrapped__ = current_turn
        progress_loop_module._generate_turn_with_context_recovery = generate_turn_with_host_verifier

    current_collector = java_lsp_trace_module._collect_diagnostics_traced
    if not getattr(current_collector, _JDT_COLLECTOR_MARKER, False):
        setattr(_collect_diagnostics_progress_aware, _JDT_COLLECTOR_MARKER, True)
        _collect_diagnostics_progress_aware.__wrapped__ = current_collector
        java_lsp_trace_module._collect_diagnostics_traced = _collect_diagnostics_progress_aware


__all__ = [
    "host_jdt_hard_timeout_seconds",
    "host_jdt_idle_timeout_seconds",
    "install",
    "run_generation_verifier",
    "synthesized_verifier_turn",
]
