"""Colab entry point for the forced native MTP llama-server runtime."""

from __future__ import annotations

from typing import Any

from . import llama_server_autotune as _server

_DEFAULT_MTP_WIDTH = 3
_FORCED_MTP_WIDTH: int | None = None


def _mtp_width() -> int:
    """Return the explicitly requested MTP draft width, defaulting to three."""

    return _server._env_int("MMM_LLAMA_MTP_WIDTH", _DEFAULT_MTP_WIDTH)


def start_colab_mtp_server(config: Any) -> str:
    """Start the managed llama-server in MTP mode without probing a baseline."""

    global _FORCED_MTP_WIDTH

    width = _mtp_width()
    variant = _server.ServerVariant(
        name=f"mtp-{width}",
        spec_type="draft-mtp",
        draft_n_max=width,
    )

    with _server._AUTOTUNE_LOCK:
        process = _server._MANAGED_PROCESS
        if (
            process is not None
            and process.poll() is None
            and _server._MANAGED_URL
            and _FORCED_MTP_WIDTH == width
        ):
            endpoint = _server._MANAGED_URL
        else:
            # Never keep a previously selected baseline managed server alive.
            _server._shutdown_managed_server()

            binary = _server._server_binary()
            if binary is None:
                raise RuntimeError("native llama-server binary is unavailable")

            model_path = _server._resolve_model_path(config)
            endpoint = _server._launch_selected(
                binary,
                model_path,
                config,
                variant,
            )
            _FORCED_MTP_WIDTH = width

    extra = getattr(config, "extra", None)
    if isinstance(extra, dict):
        extra["llama_server_url"] = endpoint
        extra["managed_llama_server"] = True
        extra["llama_server_spec_type"] = "draft-mtp"
        extra["llama_server_mtp_width"] = width

    return endpoint


__all__ = ["start_colab_mtp_server"]
