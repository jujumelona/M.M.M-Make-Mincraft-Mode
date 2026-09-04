from __future__ import annotations

"""Exact live llama.cpp chat-context accounting.

The live server remains authoritative for chat-template tokenization and slot context.
Managed-server ``n_ctx`` is cached only for the exact process generation identity;
input-token counts are always measured for the concrete payload.
"""

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_CONTEXT_CACHE_LOCK = threading.RLock()
_MANAGED_CONTEXT_CACHE: dict[str, int] = {}
# A model-facing tool call needs enough decode room to serialize one complete function
# action. The generic generation budget already enforces this floor, but exact live
# context accounting is a later authority and must never squeeze that valid request back
# down to the 0/1-token fragment that caused the coder continuation loop.
_MIN_TOOL_OUTPUT_RESERVE = 128


@dataclass(frozen=True)
class LiveContextAccounting:
    input_tokens: int
    context_tokens: int

    @property
    def remaining_tokens(self) -> int:
        return self.context_tokens - self.input_tokens


class ExactContextOverflow(RuntimeError):
    pass


def _client(server_url: str) -> Any:
    from .llama_stream_efficiency_contract import _client as managed_client

    return managed_client(server_url)


def _managed_generation_identity(server_url: str) -> str:
    try:
        from .llama_server_autotune import managed_server_generation_identity

        return managed_server_generation_identity(server_url)
    except Exception:  # noqa: BLE001 - external/unmanaged servers are intentionally uncached
        return ""


def _read_context_tokens(client: Any, server_url: str) -> int:
    origin = server_url.rstrip("/").removesuffix("/v1")
    props_response = client.get(f"{origin}/props")
    props_response.raise_for_status()
    props = props_response.json()
    settings = props["default_generation_settings"]
    context_tokens = int(settings["n_ctx"])
    if context_tokens <= 0:
        raise RuntimeError(f"llama.cpp returned invalid n_ctx={context_tokens}")
    return context_tokens


def _context_tokens(client: Any, server_url: str) -> int:
    generation = _managed_generation_identity(server_url)
    if not generation:
        return _read_context_tokens(client, server_url)
    with _CONTEXT_CACHE_LOCK:
        cached = _MANAGED_CONTEXT_CACHE.get(generation)
        if cached is not None:
            return cached
        context_tokens = _read_context_tokens(client, server_url)
        # MMM owns at most one managed llama-server process. Retaining only the live
        # generation prevents a stale process configuration from surviving restart.
        _MANAGED_CONTEXT_CACHE.clear()
        _MANAGED_CONTEXT_CACHE[generation] = context_tokens
        return context_tokens


def live_context_accounting(
    server_url: str,
    payload: Mapping[str, Any],
) -> LiveContextAccounting:
    client = _client(server_url)
    body = dict(payload)
    body.pop("stream", None)
    token_response = client.post(
        f"{server_url.rstrip('/')}/chat/completions/input_tokens",
        json=body,
    )
    token_response.raise_for_status()
    token_payload = token_response.json()
    input_tokens = int(token_payload["input_tokens"])
    context_tokens = _context_tokens(client, server_url)
    if input_tokens < 0:
        raise RuntimeError(
            "llama.cpp returned invalid live context accounting: "
            f"input_tokens={input_tokens} n_ctx={context_tokens}"
        )
    return LiveContextAccounting(
        input_tokens=input_tokens,
        context_tokens=context_tokens,
    )


def capacity_safe_payload(
    server_url: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    accounting = live_context_accounting(server_url, result)
    remaining = accounting.remaining_tokens
    if remaining <= 0:
        raise ExactContextOverflow(
            "exact llama.cpp chat input exceeds the active runtime slot; "
            f"input_tokens={accounting.input_tokens} n_ctx={accounting.context_tokens}"
        )

    raw_tools = result.get("tools")
    has_tools = bool(isinstance(raw_tools, (list, tuple)) and raw_tools)
    if has_tools and remaining < _MIN_TOOL_OUTPUT_RESERVE:
        # This is context pressure, not output exhaustion. Raising the canonical typed
        # boundary here lets the progress-aware owner compact observations *before*
        # inference instead of sending max_tokens=1, receiving finish_reason=length,
        # and entering assistant-prefill/outer-continuation recovery.
        from .llama_finish_reason_contract import (
            CONTEXT_PRESSURE,
            LlamaCompletionBoundaryError,
        )

        raise LlamaCompletionBoundaryError(
            "exact llama.cpp live context cannot fit one complete tool action; compact "
            "observations before inference; "
            f"input_tokens={accounting.input_tokens} n_ctx={accounting.context_tokens} "
            f"remaining_tokens={remaining} required_output_tokens={_MIN_TOOL_OUTPUT_RESERVE}",
            kind=CONTEXT_PRESSURE,
            prompt_tokens=accounting.input_tokens,
            max_tokens=remaining,
        )

    requested = int(result.get("max_tokens", 0) or 0)
    if requested <= 0 or requested > remaining:
        result["max_tokens"] = remaining
    return result


__all__ = [
    "ExactContextOverflow",
    "LiveContextAccounting",
    "capacity_safe_payload",
    "live_context_accounting",
]
