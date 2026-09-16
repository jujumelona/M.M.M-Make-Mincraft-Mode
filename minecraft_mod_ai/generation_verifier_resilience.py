from __future__ import annotations

"""Host-owned incremental Java generation verification.

JDT Core owns incremental dependency analysis and completed-build diagnostics.
The host owns the requested verifier deadline and mutation/model identity. JDT remains
the primary verifier; when its infrastructure is unavailable, the host may fall back
to a real Gradle build. Dependency-resolution diagnostics are corroborated with the
pinned Gradle build so a broken JDT classpath is not mistaken for a source defect.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from functools import wraps
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause

_MARKER = "_mmm_host_owned_generation_verifier"
_VERIFIER_NAME = "java_diagnostics"
_JDT_SERVICE_ATTR = "_mmm_generation_java_service"
_EXTERNAL_CODE_PREFIXES = ("net.minecraft.", "net.fabricmc.")
_DEPENDENCY_FAILURE_MARKERS = (
    "cannot be resolved",
    "does not exist",
    "is not accessible",
)


def _bounded_int_env(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def host_jdt_idle_timeout_seconds() -> int:
    """Default generation-verifier deadline when a call does not provide one."""

    return _bounded_int_env(
        "MMM_JDT_DIAGNOSTIC_IDLE_TIMEOUT_SECONDS",
        default=90,
        minimum=15,
        maximum=300,
    )


def _requested_timeout_seconds(payload: Mapping[str, Any]) -> float:
    raw = payload.get("timeout_seconds", host_jdt_idle_timeout_seconds())
    if isinstance(raw, bool):
        raise ValueError("JDT diagnostics timeout must be a positive number.")
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("JDT diagnostics timeout must be a positive number.") from exc
    if timeout <= 0 or timeout != timeout or timeout == float("inf"):
        raise ValueError("JDT diagnostics timeout must be a positive finite number.")
    return timeout


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


def _diagnostic_items(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = result.get("diagnostics")
    if isinstance(raw, Mapping):
        return tuple(
            item
            for group in raw.values()
            if isinstance(group, list)
            for item in group
            if isinstance(item, Mapping)
        )
    if isinstance(raw, list):
        return tuple(item for item in raw if isinstance(item, Mapping))
    return ()


def _is_error_diagnostic(item: Mapping[str, Any]) -> bool:
    try:
        return int(item.get("severity", 1)) == 1
    except (TypeError, ValueError, OverflowError):
        return True


def _jdt_dependency_resolution_suspect(result: Mapping[str, Any]) -> bool:
    """Detect JDT errors that may be project-classpath failures rather than bad source."""

    for item in _diagnostic_items(result):
        if not _is_error_diagnostic(item):
            continue
        message = " ".join(str(item.get("message") or "").casefold().split())
        if not any(prefix in message for prefix in _EXTERNAL_CODE_PREFIXES):
            continue
        if any(marker in message for marker in _DEPENDENCY_FAILURE_MARKERS):
            return True
    return False


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


def _structured_verifier_result(
    result: Mapping[str, Any],
    *,
    runtime_module: Any,
) -> dict[str, Any]:
    """Sanitize a verifier receipt without destroying its machine-readable schema."""

    sanitized = runtime_module._sanitize_observation(result)
    if not isinstance(sanitized, Mapping):
        raise runtime_module.AgentToolRuntimeError(
            "Generation verifier returned a non-mapping diagnostic receipt"
        )
    payload = dict(sanitized)
    payload["_mmm_observation"] = {
        "trust": "untrusted_data_only",
        "sanitized": True,
        "truncated": False,
    }
    return payload


def _run_gradle_corroboration(
    runtime: Any,
    root: Path,
    *,
    runtime_module: Any,
    jdt_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Use Gradle to distinguish source defects from a JDT dependency-classpath defect."""

    from .generation_verifier_fallback_installation import _gradle_fallback_receipt

    reason = RuntimeError(
        "JDT reported unresolved Minecraft/Fabric dependency symbols; "
        "corroborating with the pinned Gradle compile"
    )
    emit_root_cause(
        "generation_verifier_gradle_corroboration_start",
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result="START",
        reason=str(reason),
        details={"jdt_error_count": jdt_result.get("error_count")},
    )
    try:
        corroborated = _gradle_fallback_receipt(
            runtime,
            root,
            runtime_module=runtime_module,
            jdt_error=reason,
            corroboration=True,
        )
    except Exception as exc:
        emit_root_cause(
            "generation_verifier_gradle_corroboration_unavailable",
            stage="generation",
            operation="run_gradle_build",
            gate="target_compile",
            result="SKIP",
            reason=f"{type(exc).__name__}: {exc}",
            exc=exc,
        )
        return _structured_verifier_result(jdt_result, runtime_module=runtime_module)

    status = str(corroborated.get("status") or "").strip().upper()
    if status == "UNAVAILABLE":
        # JDT did return a complete diagnostic receipt. If the independent Gradle
        # cross-check cannot run, retain that real evidence instead of converting a
        # source diagnostic into a verifier-health failure.
        emit_root_cause(
            "generation_verifier_gradle_corroboration_unavailable",
            stage="generation",
            operation="run_gradle_build",
            gate="target_compile",
            result="SKIP",
            reason="Gradle corroboration unavailable; retaining complete JDT diagnostics",
            details={"gradle_result": corroborated},
        )
        return _structured_verifier_result(jdt_result, runtime_module=runtime_module)
    return corroborated


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
        requested_timeout = _requested_timeout_seconds(payload)
        service = getattr(runtime, _JDT_SERVICE_ATTR, None)
        if service is None:
            service = (java_service_factory or JavaCoreService)()
            setattr(runtime, _JDT_SERVICE_ATTR, service)
        result = service.diagnostics(
            root,
            timeout_seconds=requested_timeout,
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
    if result.get("error_count") and _jdt_dependency_resolution_suspect(result):
        return _run_gradle_corroboration(
            runtime,
            Path(root),
            runtime_module=runtime_module,
            jdt_result=result,
        )
    return _structured_verifier_result(result, runtime_module=runtime_module)


setattr(run_generation_verifier, "_mmm_generation_gradle_fallback", True)


def install(
    *,
    agent_tool_runtime_module: Any,
    progress_loop_module: Any,
    java_lsp_trace_module: Any,
) -> None:
    """Install the host-owned generation verifier after runtime composition.

    ``java_lsp_trace_module`` remains in the signature for composition compatibility,
    but generation verification no longer mutates its diagnostics collector.
    """

    del java_lsp_trace_module
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


__all__ = [
    "host_jdt_idle_timeout_seconds",
    "install",
    "run_generation_verifier",
    "synthesized_verifier_turn",
]
