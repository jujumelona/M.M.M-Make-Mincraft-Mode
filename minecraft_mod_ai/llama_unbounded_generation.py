from __future__ import annotations

"""Use llama.cpp unlimited completion only for non-tool production turns.

Tool-capable agent turns must keep the bounded budget already chosen by the base
llama.cpp payload. A tool action is a short control response, not a free-form answer;
forcing ``max_tokens=-1`` there can turn one agent step into minutes of unnecessary
decode. Non-tool production turns may still use native unlimited prediction until EOS
or the real model/server context boundary. Registry-declared bounded Qwen MTP pages
keep their stricter transport-owned limit as before.
"""

from functools import wraps
from typing import Any, Mapping

_MARKER = "_mmm_unbounded_llama_completion_v3"


def _bounded_qwen_mtp(config: Any) -> bool:
    extra = getattr(config, "extra", {})
    metadata = extra if isinstance(extra, Mapping) else {}
    return (
        str(metadata.get("runtime_contract", "")).strip().casefold() == "qwen"
        and str(metadata.get("decode_hotpath", "")).strip().casefold() == "t4_mtp"
    )


def _has_tools(request: Any) -> bool:
    return bool(getattr(request, "tools", None))


def install(hardware_module: Any) -> None:
    current = hardware_module._server_payload
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def unbounded_server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = dict(current(adapter, request))
        # Preserve the base request budget for every tool-capable turn. The model only
        # needs enough output to choose/call a tool (or finish), so unlimited decoding
        # is both unnecessary and a liveness risk on small GPUs such as T4.
        if _has_tools(request):
            return payload
        # Qwen T4/MTP owns its output policy separately. Do not erase that boundary.
        if not _bounded_qwen_mtp(getattr(adapter, "config", None)):
            payload["max_tokens"] = -1
        return payload

    setattr(unbounded_server_payload, _MARKER, True)
    hardware_module._server_payload = unbounded_server_payload


__all__ = ["install"]
