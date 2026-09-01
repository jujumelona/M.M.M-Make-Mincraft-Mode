from __future__ import annotations

"""Exact live llama.cpp chat-context accounting.

The managed server is the authority for both chat-template tokenization and the
per-slot context size.  No byte/token ratio or fixed safety margin participates in
the fit decision.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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

    origin = server_url.rstrip("/").removesuffix("/v1")
    props_response = client.get(f"{origin}/props")
    props_response.raise_for_status()
    props = props_response.json()
    settings = props["default_generation_settings"]
    context_tokens = int(settings["n_ctx"])
    if input_tokens < 0 or context_tokens <= 0:
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
