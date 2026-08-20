from __future__ import annotations

"""Use llama.cpp's native unlimited completion budget for production turns.

llama-server accepts ``max_tokens``/``n_predict`` = -1 to mean unlimited prediction
until EOS or the real model/server context boundary.  Production requests must not be
silently truncated by a registry ``max_new_tokens`` value intended for other adapters.
"""

from functools import wraps
from typing import Any

_MARKER = "_mmm_unbounded_llama_completion_v1"


def install(hardware_module: Any) -> None:
    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def unbounded_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = dict(current(adapter, request))
        # llama.cpp documents -1 as infinite n_predict/max_tokens. The actual model
        # context window and EOS remain the hard bounds; no MMM-specific 8K/16K cap.
        payload["max_tokens"] = -1
        return payload

    setattr(unbounded_server_payload, _MARKER, True)
    hardware_module._server_payload = unbounded_server_payload


__all__ = ["install"]
