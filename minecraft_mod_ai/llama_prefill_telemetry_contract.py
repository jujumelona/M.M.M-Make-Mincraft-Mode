from __future__ import annotations

from functools import wraps
from typing import Any

_MARKER = "_mmm_prompt_prefill_telemetry_v1"


def install(hardware_policy_module: Any) -> None:
    # These worker-5 runtime policies are installed after the native tuning/stream
    # pipeline exists, so no shared runtime-bootstrap owner is needed here.
    from . import (
        forced_tool_execution_contract,
        llama_decode_speed_contract,
        llama_stream_efficiency_contract,
        model_context_budget,
    )
    from .llama_completion_liveness_contract import install as install_completion_liveness
    from .llama_context_safety_contract import install as install_context_safety
    from .llama_forced_tool_capability_contract import (
        install as install_forced_tool_capability,
    )
    from .llama_kv_correctness_contract import install as install_kv_correctness
    from .model_adapters import llama_cpp_adapter

    install_completion_liveness(llama_stream_efficiency_contract, llama_cpp_adapter)
    install_kv_correctness(llama_decode_speed_contract)
    install_context_safety(model_context_budget)
    # forced_tool_execution.install runs immediately after this prefill hook. Patching
    # its module-level probe owners now makes the later adapter wrapper capture the
    # recoverable capability policy without adding a second bootstrap stage.
    install_forced_tool_capability(forced_tool_execution_contract)

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
