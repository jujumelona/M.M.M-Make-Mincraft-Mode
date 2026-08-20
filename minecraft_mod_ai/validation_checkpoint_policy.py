from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

_VALIDATION_CHECKPOINTS = frozenset({"validate-source", "validate-jdt"})


def _file_digest(module: Any) -> str:
    path_value = getattr(module, "__file__", "")
    if not path_value:
        return "missing"
    try:
        return hashlib.sha256(Path(path_value).resolve().read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def validation_implementation_fingerprint(checkpoint_id: str) -> str:
    """Hash the validation implementation and active MMM runtime policy.

    Validation checkpoints are reusable only when their generated inputs, validation
    implementation, and host policy all match the original successful run.
    """

    if checkpoint_id not in _VALIDATION_CHECKPOINTS:
        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")

    from . import complete_orchestrator, java_lsp, scalable_validator, scale_policy, validator

    modules = [complete_orchestrator]
    if checkpoint_id == "validate-source":
        modules.extend((scalable_validator, validator, scale_policy))
    else:
        modules.append(java_lsp)

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


def validation_checkpoint_input(
    checkpoint_id: str,
    input_value: Mapping[str, Any],
) -> dict[str, Any]:
    scoped = dict(input_value)
    scoped["_mmm_validation_implementation"] = validation_implementation_fingerprint(
        checkpoint_id
    )
    return scoped


def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
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


__all__ = [
    "cached_validation_is_reusable",
    "validation_checkpoint_input",
    "validation_implementation_fingerprint",
]
