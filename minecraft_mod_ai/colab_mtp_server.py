"""Colab entry point for the single native llama-server runtime."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .llama_server_autotune import ensure_tuned_server


def start_colab_mtp_server(config: Any) -> str:
    """Start or reuse the tuned native llama-server used by the Colab notebook."""

    benchmark_request = SimpleNamespace(messages=(), response_format="text")
    return ensure_tuned_server(config, benchmark_request)


__all__ = ["start_colab_mtp_server"]
