"""Compatibility entry point for the Colab managed llama-server.

The historical module name is kept because the canonical notebook imports it, but
this module no longer forces MTP.  Runtime selection belongs to the native llama
autotuner, which compares baseline and speculative variants on the active model
and hardware before launching the persistent server.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .llama_server_autotune import ensure_tuned_server


def start_colab_mtp_server(config: Any) -> str:
    """Start the single managed server using the measured decode-speed winner."""

    request = SimpleNamespace(messages=(), response_format="text")
    return ensure_tuned_server(config, request)


__all__ = ["start_colab_mtp_server"]
