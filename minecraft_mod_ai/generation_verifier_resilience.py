from __future__ import annotations

"""Host-owned incremental Java generation verification.

JDT Core owns incremental dependency analysis and completed-build diagnostics.
The host owns deadlines and mutation/model identity. No inferred unchanged PASS,
source hash scans, alternate backend, or run-local circuit breaker is used.
"""

import hashlib
import json
import os
import queue
import time
from collections import Counter
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause

_MARKER = "_mmm_host_owned_generation_verifier"
_JDT_COLLECTOR_MARKER = "_mmm_progress_aware_jdt_collector"
_VERIFIER_NAME = "java_diagnostics"
_JDT_SERVICE_ATTR = "_mmm_generation_java_service"


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
    identity_material = f"{len(messages)}:{raw_arguments}".encode()
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


def _close_generation_jdt(runtime: Any) -> None:
    service = getattr(runtime, _JDT_SERVICE_ATTR, None)
    try:
        if service is not None:
            service.close()
    finally:
        if hasattr(runtime, _JDT_SERVICE_ATTR):
            delattr(runtime, _JDT_SERVICE_ATTR)


def _run_gradle_fallback(
    runtime: Any,
    root: Path,
    *,
    runtime_module: Any,
    jdt_error: BaseException,
) -> dict[str, Any]:
    from .generation_verifier_fallback_installation import _gradle_fallback_receipt

    emit_root_cause(
        "generation_verifier_gradle_fallback_start",
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result="START",
        reason=str(jdt_error),
    )
    try:
        return _gradle_fallback_receipt(
            runtime,
            root,
            runtime_module=runtime_module,
            jdt_error=jdt_error,
        )
    except Exception as fallback_exc:
        emit_root_cause(
            "generation_verifier_gradle_fallback_unavailable",
            stage="generation",
            operation="run_gradle_build",
            gate="target_compile",
            result="FAIL",
            reason=f"{type(fallback_exc).__name__}: {fallback_exc}",
            exc=fallback_exc,
        )
        raise runtime_module.AgentToolRuntimeError(
            "Generation verification has no healthy backend: "
            f"JDT unavailable ({jdt_error}); Gradle fallback unavailable "
            f"({type(fallback_exc).__name__}: {fallback_exc})"
        ) from fallback_exc


def run_generation_verifier(
    runtime: Any,
    arguments: Mapping[str, Any] | None,
    *,
    runtime_module: Any,
    java_service_factory: Any | None = None,
) -> dict[str, Any]:
    """Return one completed host verifier result for the current project state."""
    from .java_core import JavaCoreService
    from .java_lsp import JDTLanguageServerError
    from .owner_rpc import OwnerRPCError

    payload = dict(arguments or {})
    root, _ = runtime_module._discover_model_project_root(runtime.workspace_root)
    try:
        _normalize_relative_files(payload.get("relative_files"))
        service = getattr(runtime, _JDT_SERVICE_ATTR, None)
        if service is None:
            service = (java_service_factory or JavaCoreService)()
            setattr(runtime, _JDT_SERVICE_ATTR, service)
        result = service.diagnostics(
            root,
            timeout_seconds=host_jdt_hard_timeout_seconds(host_jdt_idle_timeout_seconds()),
            full_scan=bool(payload.get("full_scan", False)),
        )
        if (
            result.get("complete") is not True
            or not result.get("session_id")
            or not result.get("model_id")
        ):
            raise OwnerRPCError("JDT Core returned incomplete or unbound verification")
    except (
        OSError,
        ValueError,
        TypeError,
        TimeoutError,
        JDTLanguageServerError,
        OwnerRPCError,
    ) as exc:
        _close_generation_jdt(runtime)
        emit_root_cause(
            "generation_verifier_jdt_unavailable",
            stage="generation",
            operation=_VERIFIER_NAME,
            result="FAIL",
            reason=str(exc),
            exc=exc,
        )
        return _run_gradle_fallback(
            runtime,
            Path(root),
            runtime_module=runtime_module,
            jdt_error=exc,
        )

    emit_root_cause(
        "generation_verifier_jdt_result",
        stage="generation",
        operation=_VERIFIER_NAME,
        result="FAIL" if result.get("error_count") else "PASS",
        details={"result": result},
    )
    return runtime_module._bounded_result(result)


setattr(run_generation_verifier, "_mmm_generation_gradle_fallback", True)


def _raise_diagnostic_timeout(
    rpc: Any,
    *,
    expected_uris: set[str],
    diagnostics: dict[str, list[dict[str, Any]]],
    unexpected_uris: set[str],
    ignored_methods: Counter[str],
    malformed_messages: int,
    started: float,
    last_progress: float,
    page_index: int,
    timeout_kind: str,
    timeout_seconds: float,
    hard_timeout: float,
) -> None:
    from .java_lsp import JDTLanguageServerError

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
    from .java_lsp import (
        JDTLanguageServerError,
        _respond_to_server_request,
        _sorted_diagnostics,
    )

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
            wait_seconds = min(
                wait_seconds,
                max(0.001, quiet_seconds - (now - settled_since)),
            )
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

    _raise_diagnostic_timeout(
        rpc,
        expected_uris=expected_uris,
        diagnostics=diagnostics,
        unexpected_uris=unexpected_uris,
        ignored_methods=ignored_methods,
        malformed_messages=malformed_messages,
        started=started,
        last_progress=last_progress,
        page_index=page_index,
        timeout_kind=timeout_kind,
        timeout_seconds=timeout_seconds,
        hard_timeout=hard_timeout,
    )
    raise AssertionError("unreachable")


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
            if (
                selected == "generation"
                and str(name).strip() == _VERIFIER_NAME
                and external_server_ids is None
            ):
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
                    reason=(
                        "forced verifier selection is mechanical and does not require "
                        "coder inference"
                    ),
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
        progress_loop_module._generate_turn_with_context_recovery = (
            generate_turn_with_host_verifier
        )

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
