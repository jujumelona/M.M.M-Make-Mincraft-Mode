from __future__ import annotations

"""Keep assistant-prefill calibration strict while preserving outer recovery.

``llama_finish_reason_contract`` classifies output exhaustion and the custom-module
checkpoint owner can resume a failed bounded action.  That does not require changing
the calibration API itself.  In particular, callers and tests must still observe a
real zero-token probe and must still receive an error when the server generates tokens,
reasoning, tool calls, or otherwise returns an ambiguous template prefix.

An earlier resilience wrapper swallowed those errors globally and skipped the probe on
tool turns.  This contract removes only that wrapper.  The typed OUTPUT_EXHAUSTED error
and preserved partial message remain available to the outer checkpoint continuation.
"""

from typing import Any

_MARKER = "_mmm_nonfatal_prefill_calibration"


def install(llama_cpp_module: Any) -> None:
    current = getattr(
        llama_cpp_module,
        "_calibrate_assistant_prefill_generation_prompt",
        None,
    )
    if not callable(current) or not getattr(current, _MARKER, False):
        return
    original = getattr(current, "__wrapped__", None)
    if not callable(original):
        raise RuntimeError(
            "non-fatal prefill calibration wrapper lost its strict wrapped function"
        )
    llama_cpp_module._calibrate_assistant_prefill_generation_prompt = original


__all__ = ["install"]
