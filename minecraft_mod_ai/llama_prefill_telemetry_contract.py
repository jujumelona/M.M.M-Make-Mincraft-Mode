from __future__ import annotations

from functools import wraps
from typing import Any

_MARKER = "_mmm_prompt_prefill_telemetry_v1"


def install(hardware_policy_module: Any) -> None:
    # llama_stream_efficiency is installed immediately before this contract. Keep
    # completion liveness as a worker-5 runtime concern without adding another shared
    # runtime-bootstrap owner.
    from . import llama_stream_efficiency_contract
    from .llama_completion_liveness_contract import install as install_completion_liveness

    install_completion_liveness(llama_stream_efficiency_contract)

    current = hardware_policy_module._commit_metrics_delta
    if getattr(current, _MARKER, False):
        return

    with hardware_policy_module._TELEMETRY_LOCK:
        hardware_policy_module._TELEMETRY_TOTALS.setdefault("prompt_seconds", 0.0)

    @wraps(current)
    def commit_with_prefill(before, after):
        result = current(before, after)
        if result is None or before is None or after is None:
            return result

        prompt_seconds = max(
            0.0,
            float(after.get("prompt_seconds_total", 0.0))
            - float(before.get("prompt_seconds_total", 0.0)),
        )
        prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
        prompt_tps = prompt_tokens / prompt_seconds if prompt_seconds > 0 else 0.0
        with hardware_policy_module._TELEMETRY_LOCK:
            totals = hardware_policy_module._TELEMETRY_TOTALS
            totals["prompt_seconds"] = float(totals.get("prompt_seconds", 0.0)) + prompt_seconds
            cumulative_prompt_seconds = float(totals["prompt_seconds"])
            cumulative_prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
        cumulative_prompt_tps = (
            cumulative_prompt_tokens / cumulative_prompt_seconds
            if cumulative_prompt_seconds > 0
            else 0.0
        )
        enriched = dict(result)
        enriched.update(
            {
                "prompt_seconds": prompt_seconds,
                "prompt_tps": prompt_tps,
                "cumulative_prompt_seconds": cumulative_prompt_seconds,
                "cumulative_prompt_tps": cumulative_prompt_tps,
            }
        )
        print(
            "llama server: prefill complete",
            f" prompt_tokens={prompt_tokens}",
            f" prompt_seconds={prompt_seconds:.3f}",
            f" prompt_tok_s={prompt_tps:.2f}",
            f" cumulative_prompt_tok_s={cumulative_prompt_tps:.2f}",
            sep="",
            flush=True,
        )
        return enriched

    setattr(commit_with_prefill, _MARKER, True)
    commit_with_prefill.__wrapped__ = current  # type: ignore[attr-defined]
    hardware_policy_module._commit_metrics_delta = commit_with_prefill


__all__ = ["install"]
