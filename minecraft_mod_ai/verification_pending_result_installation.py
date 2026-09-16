from __future__ import annotations

"""Expose verifier-unavailable generation as an explicit resumable API.

The normal ``CustomModuleGenerator.generate`` contract remains fail-closed for production.
Callers that intentionally support pause/resume call ``generate_resumable`` explicitly;
no class method is injected at package import time. Only verifier-infrastructure exhaustion
is converted to a pending receipt. Source defects, protocol errors and every unrelated
configuration error still raise normally.
"""

from typing import Any

from .model_adapters import ModelConfigurationError

_PENDING_CODES = (
    "VERIFIER_UNAVAILABLE",
    "VERIFIER_RECOVERY_UNAVAILABLE",
)


def _pending_reason_code(exc: BaseException) -> str | None:
    text = str(exc or "").strip()
    for code in _PENDING_CODES:
        if text == code or text.startswith(code + ":"):
            return code
    return None


def _required_gates(module: Any) -> list[str]:
    raw = getattr(module, "required_gates", ()) if module is not None else ()
    values = ["JDT", "Gradle", "GameTest"]
    try:
        values.extend(str(item).strip() for item in raw if str(item).strip())
    except TypeError:
        pass
    return list(dict.fromkeys(values))


def _verification_pending_result(module: Any, exc: BaseException) -> dict[str, Any]:
    reason_code = _pending_reason_code(exc)
    if reason_code is None:
        raise ValueError("verification pending result requires a verifier-unavailable error")
    return {
        "schema_version": "mmm/verification-pending-v1",
        "module_id": str(getattr(module, "module_id", "") or ""),
        "kind": str(getattr(module, "kind", "") or ""),
        "status": "VERIFICATION_PENDING",
        "resumable": True,
        "release_ready": False,
        "verification": {
            "status": "UNAVAILABLE",
            "phase": "VERIFY",
            "verified": False,
            "reason_code": reason_code,
            "detail": str(exc),
        },
        "generation_checkpoint": {
            "status": "PRESERVED_FOR_RESUME",
            "phase": "VERIFY",
            "resume_action": "retry_same_custom_module_generation",
        },
        "required_gates": _required_gates(module),
    }


def generate_resumable(generator: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run one generation while converting only verifier-unavailable failures to pending."""

    try:
        return generator.generate(*args, **kwargs)
    except ModelConfigurationError as exc:
        if _pending_reason_code(exc) is None:
            raise
        return _verification_pending_result(kwargs.get("module"), exc)


__all__ = [
    "_pending_reason_code",
    "_verification_pending_result",
    "generate_resumable",
]
