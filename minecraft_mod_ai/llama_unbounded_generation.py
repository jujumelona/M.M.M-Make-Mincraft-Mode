from __future__ import annotations

"""Use llama.cpp's native unlimited completion budget for production turns.

llama-server accepts ``max_tokens``/``n_predict`` = -1 to mean unlimited prediction
until EOS or the real model/server context boundary. Production requests must not be
silently truncated by a registry ``max_new_tokens`` value intended for other adapters.
Registry-declared bounded Qwen MTP pages keep their stricter transport-owned limit.
"""

from functools import wraps
from typing import Any, Mapping

_MARKER = "_mmm_unbounded_llama_completion_v2"


def _bounded_qwen_mtp(config: Any) -> bool:
    extra = getattr(config, "extra", {})
    metadata = extra if isinstance(extra, Mapping) else {}
    return (
        str(metadata.get("runtime_contract", "")).strip().casefold() == "qwen"
        and str(metadata.get("decode_hotpath", "")).strip().casefold() == "t4_mtp"
    )


def install(hardware_module: Any) -> None:
    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def unbounded_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = dict(current(adapter, request))
        # The Qwen T4/MTP output policy owns its bounded page/section budget. Do not
        # erase that liveness boundary after composition. Other llama.cpp turns keep
        # native unlimited prediction until EOS or the actual context boundary.
        if not _bounded_qwen_mtp(getattr(adapter, "config", None)):
            payload["max_tokens"] = -1
        return payload

    setattr(unbounded_server_payload, _MARKER, True)
    hardware_module._server_payload = unbounded_server_payload


__all__ = ["install"]
