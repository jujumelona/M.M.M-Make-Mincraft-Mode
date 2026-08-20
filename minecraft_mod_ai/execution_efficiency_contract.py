from __future__ import annotations

import hashlib
import os
from functools import wraps
from pathlib import Path
from typing import Any


_VALIDATION_CHECKPOINTS = frozenset({"validate-source", "validate-jdt"})


def _literal_false_validator(callback: Any) -> bool:
    """Recognize the legacy `lambda _cached: False` without overriding real policy."""

    code = getattr(callback, "__code__", None)
    if code is None or code.co_freevars or code.co_names:
        return False
    constants = tuple(value for value in code.co_consts if value is not None)
    return constants == (False,)


def _file_digest(module: Any) -> str:
    path_value = getattr(module, "__file__", "")
    if not path_value:
        return "missing"
    try:
        return hashlib.sha256(Path(path_value).resolve().read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _validation_implementation_fingerprint(checkpoint_id: str) -> str:
    """Fingerprint live validation code plus MMM runtime policy, not project output.

    Persistent checkpoint reuse is safe only when both generated inputs and validation
    semantics are unchanged. Hashing the relevant implementation files and all MMM_*
    policy variables makes a checkout/configuration change an automatic cache miss.
    Raw environment values are never persisted; only this digest enters the checkpoint
    input hash.
    """

    from . import complete_orchestrator, java_lsp, scalable_validator, scale_policy, validator

    modules = [complete_orchestrator]
    if checkpoint_id == "validate-source":
        modules.extend((scalable_validator, validator, scale_policy))
    elif checkpoint_id == "validate-jdt":
        modules.append(java_lsp)
    else:
        return ""

    digest = hashlib.sha256()
    for module in modules:
        digest.update(str(getattr(module, "__name__", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(module).encode("ascii"))
        digest.update(b"\0")
    for name, value in sorted(
        (name, value)
        for name, value in os.environ.items()
        if name.startswith("MMM_")
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"=")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if checkpoint_id == "validate-source":
        return value.get("status") == "PASS"
    if checkpoint_id == "validate-jdt":
        return (
            value.get("schema_version") == "mmm/java-diagnostics-v2"
            and isinstance(value.get("diagnostics"), dict)
        )
    return False


def _install_validation_checkpoint_reuse(orchestrator_module: Any) -> None:
    current = orchestrator_module.run_named_checkpoint
    if getattr(current, "_mmm_exact_validation_checkpoint_reuse", False):
        return

    @wraps(current)
    def run_named_checkpoint(
        ledger: Any,
        checkpoint_id: str,
        *,
        stage: str,
        input_value: Any,
        action: Any,
        encode: Any,
        decode: Any,
        validate_cached: Any = None,
    ) -> Any:
        if (
            checkpoint_id in _VALIDATION_CHECKPOINTS
            and _literal_false_validator(validate_cached)
        ):
            scoped_input = (
                dict(input_value)
                if isinstance(input_value, dict)
                else {"input_value": input_value}
            )
            scoped_input["_mmm_validation_implementation"] = (
                _validation_implementation_fingerprint(checkpoint_id)
            )
            return current(
                ledger,
                checkpoint_id,
                stage=stage,
                input_value=scoped_input,
                action=action,
                encode=encode,
                decode=decode,
                validate_cached=lambda value: _cached_validation_is_reusable(
                    checkpoint_id,
                    value,
                ),
            )
        return current(
            ledger,
            checkpoint_id,
            stage=stage,
            input_value=input_value,
            action=action,
            encode=encode,
            decode=decode,
            validate_cached=validate_cached,
        )

    run_named_checkpoint._mmm_exact_validation_checkpoint_reuse = True  # type: ignore[attr-defined]
    run_named_checkpoint.__wrapped__ = current  # type: ignore[attr-defined]
    orchestrator_module.run_named_checkpoint = run_named_checkpoint


def install(*, work_graph_module: Any | None = None) -> None:
    """Install exact-input validation resume reuse once."""

    from . import complete_orchestrator

    _install_validation_checkpoint_reuse(complete_orchestrator)


__all__ = [
    "install",
    "_cached_validation_is_reusable",
    "_install_validation_checkpoint_reuse",
    "_literal_false_validator",
    "_validation_implementation_fingerprint",
]
