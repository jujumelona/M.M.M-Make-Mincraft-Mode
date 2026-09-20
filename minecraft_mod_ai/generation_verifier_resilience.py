from __future__ import annotations

"""Host-owned incremental Java generation verification.

JDT Core is the sole generation-time source verifier. Its project-model resolution may
use Gradle Tooling API data internally, but Gradle builds are never promoted to a hidden
verification fallback or corroboration backend. Verifier infrastructure failures are
reported as structured UNAVAILABLE receipts so the host loop can make an explicit
verifier decision without re-entering a Gradle build path.
"""

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .root_cause_trace import emit_root_cause

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
    """Default generation-verifier deadline when a call does not provide one."""

    return _bounded_int_env(
        "MMM_JDT_DIAGNOSTIC_IDLE_TIMEOUT_SECONDS",
        default=90,
        minimum=15,
        maximum=300,
    )


def host_jdt_startup_timeout_seconds() -> int:
    """Hard budget for creating and importing a cold generation JDT owner."""

    return _bounded_int_env(
        "MMM_JDT_DIAGNOSTIC_STARTUP_TIMEOUT_SECONDS",
        default=300,
        minimum=90,
        maximum=600,
    )

class _VerifierWatchdogTimeout(TimeoutError):
    pass


def _watchdog_grace_seconds() -> int:
    return _bounded_int_env(
        "MMM_JDT_DIAGNOSTIC_WATCHDOG_GRACE_SECONDS",
        default=5,
        minimum=1,
        maximum=30,
    )


def _diagnostics_with_watchdog(
    service: Any,
    root: Path,
    *,
    relative_files: list[str] | None,
    timeout_seconds: float,
    full_scan: bool,
) -> Mapping[str, Any]:
    """Enforce the verifier wall-clock budget even if a lower layer wedges."""

    completed = threading.Event()
    result_box: dict[str, Any] = {}
    error_box: list[BaseException] = []

    def worker() -> None:
        try:
            result_box["result"] = service.diagnostics(
                root,
                relative_files=relative_files,
                timeout_seconds=timeout_seconds,
                full_scan=full_scan,
            )
        except BaseException as exc:  # noqa: BLE001 - preserve verifier failure
            error_box.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(
        target=worker,
        daemon=True,
        name="mmm-generation-jdt-watchdog",
    )
    thread.start()
    wall_clock_budget = float(timeout_seconds) + float(_watchdog_grace_seconds())
    if not completed.wait(wall_clock_budget):
        abort = getattr(service, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
        completed.wait(2.0)
        raise _VerifierWatchdogTimeout(
            f"JDT diagnostics wall-clock deadline exceeded after {wall_clock_budget:.1f}s"
        )
    if error_box:
        raise error_box[0]
    result = result_box.get("result")
    if not isinstance(result, Mapping):
        raise TypeError("JDT diagnostics returned a non-mapping result")
    return result


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


def synthesized_verifier_turn(
    messages: list[dict[str, Any]],
    *,
    relative_files: tuple[str, ...] | None = None,
) -> Any:
    """Create the mechanical verifier call bound to the host-owned target."""

    from .model_adapters import GenerationResponse, ToolCall

    arguments: dict[str, Any] = {
        "timeout_seconds": host_jdt_idle_timeout_seconds(),
    }
    if relative_files:
        arguments["relative_files"] = list(dict.fromkeys(relative_files))
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


def _unavailable_verifier_result(
    root: Path,
    exc: BaseException,
    *,
    runtime_module: Any,
) -> dict[str, Any]:
    result = {
        "schema_version": "mmm/java-diagnostics-v3",
        "project_root": str(root),
        "verification_backend": "jdt_core",
        "status": "UNAVAILABLE",
        "available": False,
        "complete": False,
        "skipped": True,
        "error_count": 0,
        "warning_count": 0,
        "diagnostics": [
            {
                "severity": 1,
                "code": "JDT_DIAGNOSTICS_UNAVAILABLE",
                "source": "jdt_core",
                "message": f"{type(exc).__name__}: {exc}",
            }
        ],
    }
    return _structured_verifier_result(result, runtime_module=runtime_module)


def _generation_verifier_session(
    runtime: Any,
    requested_timeout: float,
    *,
    java_service_factory: Any | None,
) -> tuple[Any, float, bool]:
    """Return the persistent verifier service and the deadline for this call."""

    service = getattr(runtime, _JDT_SERVICE_ATTR, None)
    if service is not None:
        return service, requested_timeout, False

    from .java_core import JavaCoreService

    service = (java_service_factory or JavaCoreService)()
    setattr(runtime, _JDT_SERVICE_ATTR, service)
    startup_timeout = float(host_jdt_startup_timeout_seconds())
    return service, max(requested_timeout, startup_timeout), True


def run_generation_verifier(
    runtime: Any,
    arguments: Mapping[str, Any] | None,
    *,
    runtime_module: Any,
    java_service_factory: Any | None = None,
) -> dict[str, Any]:
    """Return one completed host verifier result for the current project state."""

    from .java_lsp import JDTLanguageServerError
    from .owner_rpc import OwnerRPCError

    payload = dict(arguments or {})
    root, _ = runtime_module._discover_model_project_root(runtime.workspace_root)
    try:
        relative_files = _normalize_relative_files(payload.get("relative_files"))
        requested_timeout = _requested_timeout_seconds(payload)
        service, effective_timeout, cold_start = _generation_verifier_session(
            runtime,
            requested_timeout,
            java_service_factory=java_service_factory,
        )
        emit_root_cause(
            "generation_verifier_deadline_selected",
            stage="generation",
            operation=_VERIFIER_NAME,
            result="PASS",
            details={
                "cold_start": cold_start,
                "requested_timeout_seconds": requested_timeout,
                "effective_timeout_seconds": effective_timeout,
            },
        )
        result = _diagnostics_with_watchdog(
            service,
            Path(root),
            relative_files=relative_files,
            timeout_seconds=effective_timeout,
            full_scan=bool(payload.get("full_scan", False)),
        )
        if (
            result.get("complete") is not True
            or not result.get("session_id")
            or not result.get("model_id")
        ):
            raise OwnerRPCError("JDT Core returned incomplete or unbound verification")
    except _VerifierWatchdogTimeout as exc:
        service = getattr(runtime, _JDT_SERVICE_ATTR, None)
        if service is not None:
            try:
                delattr(runtime, _JDT_SERVICE_ATTR)
            except AttributeError:
                pass
        emit_root_cause(
            "generation_verifier_watchdog_timeout",
            stage="generation",
            operation=_VERIFIER_NAME,
            result="FAIL",
            reason=str(exc),
            exc=exc,
        )
        return _unavailable_verifier_result(
            Path(root),
            exc,
            runtime_module=runtime_module,
        )
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
        return _unavailable_verifier_result(
            Path(root),
            exc,
            runtime_module=runtime_module,
        )

    emit_root_cause(
        "generation_verifier_jdt_result",
        stage="generation",
        operation=_VERIFIER_NAME,
        result="FAIL" if result.get("error_count") else "PASS",
        details={"result": result},
    )
    return _structured_verifier_result(result, runtime_module=runtime_module)


__all__ = [
    "host_jdt_idle_timeout_seconds",
    "host_jdt_startup_timeout_seconds",
    "run_generation_verifier",
    "synthesized_verifier_turn",
]
