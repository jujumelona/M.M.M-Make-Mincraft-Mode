from __future__ import annotations

"""Make KV-cache autotuning compare candidates against the highest-precision baseline.

``llama_decode_speed_contract`` deliberately requires exact deterministic probe output
before a faster KV type is eligible. Its probing order is therefore semantically
important: the first successful candidate is the correctness reference. Operator
preference may control the eventual non-autotuned cache type, but it must not make q4
become the truth oracle during correctness-gated autotuning.
"""

import hashlib
from functools import wraps
from typing import Any, Iterable

_PROBE_MARKER = "_mmm_kv_precision_reference_v1"
_FINGERPRINT_MARKER = "_mmm_kv_precision_reference_fingerprint_v1"
_POLICY_VERSION = "mmm/kv-reference-v1-f16-q8-q4"
_PRECISION_RANK = {"q4_0": 0, "q8_0": 1, "f16": 2}


def _precision_reference_order(candidates: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(value).strip().lower() for value in candidates)
    indexed = tuple(enumerate(values))
    ordered = sorted(
        indexed,
        key=lambda item: (-_PRECISION_RANK.get(item[1], -1), item[0]),
    )
    return tuple(value for _, value in ordered)


def install(decode_speed_module: Any) -> None:
    """Make KV probing precision-referenced and invalidate pre-policy cache receipts."""

    current_fingerprint = decode_speed_module._kv_fingerprint
    if not getattr(current_fingerprint, _FINGERPRINT_MARKER, False):

        @wraps(current_fingerprint)
        def correctness_fingerprint(*args: Any, **kwargs: Any) -> str:
            previous = str(current_fingerprint(*args, **kwargs))
            return hashlib.sha256(
                f"{previous}\0{_POLICY_VERSION}".encode("utf-8")
            ).hexdigest()

        setattr(correctness_fingerprint, _FINGERPRINT_MARKER, True)
        correctness_fingerprint.__wrapped__ = current_fingerprint  # type: ignore[attr-defined]
        decode_speed_module._kv_fingerprint = correctness_fingerprint

    current_probe = decode_speed_module._probe_kv_types
    if getattr(current_probe, _PROBE_MARKER, False):
        return

    @wraps(current_probe)
    def correctness_referenced_probe(
        autotune: Any,
        binary: str,
        model_path: str,
        config: Any,
        request: Any,
        candidates: tuple[str, ...],
    ) -> Any:
        return current_probe(
            autotune,
            binary,
            model_path,
            config,
            request,
            _precision_reference_order(candidates),
        )

    setattr(correctness_referenced_probe, _PROBE_MARKER, True)
    correctness_referenced_probe.__wrapped__ = current_probe  # type: ignore[attr-defined]
    decode_speed_module._probe_kv_types = correctness_referenced_probe


__all__ = ["_precision_reference_order", "install"]
