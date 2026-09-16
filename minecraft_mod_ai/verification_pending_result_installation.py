from __future__ import annotations

"""Expose verifier-unavailable generation as an explicit resumable result.

The normal ``CustomModuleGenerator.generate`` contract remains fail-closed for production.
Callers that intentionally support pause/resume may use ``generate_resumable``; only
verifier-infrastructure exhaustion is converted to a pending receipt. Source defects,
protocol errors and every unrelated configuration error still raise normally.
"""

from functools import wraps
from typing import Any

from .model_adapters import ModelConfigurationError

_INSTALL_MARKER = "_mmm_verification_pending_result_installed"
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


def install(custom_module_generator_module: Any) -> None:
    """Add a resumable entry point without weakening the production generate contract."""

    Generator = custom_module_generator_module.CustomModuleGenerator
    if bool(getattr(Generator, _INSTALL_MARKER, False)):
        return

    production_generate = Generator.generate

    @wraps(production_generate)
    def generate_resumable(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return self.generate(*args, **kwargs)
        except ModelConfigurationError as exc:
            if _pending_reason_code(exc) is None:
                raise
            return _verification_pending_result(kwargs.get("module"), exc)

    Generator.generate_resumable = generate_resumable
    setattr(Generator, _INSTALL_MARKER, True)


__all__ = [
    "_pending_reason_code",
    "_verification_pending_result",
    "install",
]
