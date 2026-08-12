from __future__ import annotations

import os
from functools import wraps
from typing import Any


_SERVER_KEY: tuple[Any, ...] | None = None


def _server_key(server_module: Any, config: Any, mode: str) -> tuple[Any, ...]:
    effective_mode = mode
    if effective_mode == "mtp" and not server_module._mtp_capable(config):
        effective_mode = "baseline"
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", ""))
        if isinstance(extra, dict)
        else ""
    )
    return (
        str(getattr(config, "model_id", "")),
        filename,
        min(int(getattr(config, "max_context", 0) or 0), server_module.SERVER_CONTEXT_CAP),
        int(getattr(config, "max_new_tokens", 0) or 0),
        effective_mode,
        server_module._mtp_width() if effective_mode == "mtp" else 0,
        server_module._kv_cache_quant(),
    )


def install(server_module: Any) -> None:
    """Restart the managed server when a session changes decode-critical settings.

    The optional Colab server cell can run before ``CompleteModAISession`` exists.
    A later session may therefore change ``MMM_KV_CACHE_QUANT`` (or select a model
    profile with different context/output limits) while the already-running process
    still owns the old configuration.  ``start_colab_mtp_server`` previously reused
    that process solely because its baseline/MTP mode matched.

    Bind reuse to the complete decode configuration instead.  The wrapper is used by
    both direct notebook startup and ``ensure_colab_server_for_request`` because that
    function resolves ``start_colab_mtp_server`` from the module at call time.
    """

    current = server_module.start_colab_mtp_server
    if getattr(current, "_mmm_server_config_bound", False):
        return

    @wraps(current)
    def start_bound(config: Any, *, mode: str = "baseline") -> str:
        global _SERVER_KEY

        desired_key = _server_key(server_module, config, mode)
        if server_module.colab_mtp_server_running() and _SERVER_KEY != desired_key:
            server_module.stop_colab_mtp_server(keep_enabled=True)
            print(
                "llama server: decode configuration changed; restarting",
                f" mode={desired_key[4]}",
                f" kv_cache={desired_key[6]}",
                flush=True,
            )

        url = current(config, mode=mode)
        _SERVER_KEY = _server_key(
            server_module,
            config,
            server_module.current_server_mode() or mode,
        )
        return url

    start_bound._mmm_server_config_bound = True  # type: ignore[attr-defined]
    server_module.start_colab_mtp_server = start_bound


__all__ = ["install"]
