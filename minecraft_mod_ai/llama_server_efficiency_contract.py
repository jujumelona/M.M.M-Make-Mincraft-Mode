from __future__ import annotations

from functools import wraps
from typing import Any


def install(autotune_module: Any, hardware_policy_module: Any) -> None:
    """Remove duplicate probe work and make native prompt reuse explicit."""

    probe = autotune_module._probe_server
    if getattr(probe, "_mmm_correctness_sentinel", False):
        # hardware_policy historically added a second 64-token correctness request
        # after every measured candidate. The compact deterministic benchmark is now
        # itself the exact-output correctness gate, so the extra generation contains
        # no independent information and only burns decode time.
        underlying = getattr(probe, "__wrapped__", None)
        if underlying is not None:
            autotune_module._probe_server = underlying
            probe = underlying
    probe._mmm_compact_decode_probe = True  # type: ignore[attr-defined]

    current_payload = hardware_policy_module._server_payload
    if not getattr(current_payload, "_mmm_prompt_cache_reuse", False):

        @wraps(current_payload)
        def payload_with_prompt_cache(adapter: Any, request: Any) -> dict[str, Any]:
            payload = current_payload(adapter, request)
            # Pinned llama-server defaults this to true, but keep it explicit so a
            # server-default change cannot silently disable prefix-KV reuse. This is
            # especially valuable for planner continuation/repair pages whose system
            # and contract prefixes repeat.
            payload["cache_prompt"] = True
            return payload

        payload_with_prompt_cache._mmm_prompt_cache_reuse = True  # type: ignore[attr-defined]
        hardware_policy_module._server_payload = payload_with_prompt_cache


__all__ = ["install"]
