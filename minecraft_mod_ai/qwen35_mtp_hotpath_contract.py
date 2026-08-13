from __future__ import annotations

import os
from functools import wraps
from typing import Any


_DEFAULT_WIDTH = 3
_MARKER = "_mmm_qwen35_mtp3_hotpath"


def _enabled() -> bool:
    raw = os.environ.get("MMM_QWEN35_MTP_HOTPATH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _is_qwen35_mtp(config: Any) -> bool:
    model_id = str(getattr(config, "model_id", "")).casefold()
    extra = getattr(config, "extra", {})
    filename = (
        str(extra.get("gguf_filename", "")).casefold()
        if isinstance(extra, dict)
        else ""
    )
    return (
        "qwen3.5-9b" in model_id
        and ("mtp" in model_id or "mtp" in filename)
    )


def _width(autotune: Any) -> int:
    return autotune._env_int("MMM_QWEN35_MTP_WIDTH", _DEFAULT_WIDTH, minimum=1)


def install(autotune: Any) -> None:
    """Keep the known-fast Qwen3.5-9B MTP path out of baseline re-selection.

    MMM measured the local Qwen3.5-9B MTP-3 server at roughly 80 tok/s while the
    non-speculative path was roughly 30 tok/s. The generic correctness/autotune stack
    remains available for every other model and can be explicitly restored for this
    profile with MMM_QWEN35_MTP_HOTPATH=0.

    This wrapper deliberately selects one native server variant only. It does not
    create a second GGUF engine, does not alter generation payloads, and preserves the
    canonical managed-server lifecycle/export wrappers already installed underneath.
    """

    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_qwen35_mtp3(config: Any, request: Any) -> str:
        if not _enabled() or not _is_qwen35_mtp(config):
            return current(config, request)

        # Respect an explicitly supplied external server. A live MMM-managed server
        # also stays hot instead of being restarted between planner/coder calls.
        managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if managed_process is not None and managed_process.poll() is None and managed_url:
            return managed_url
        if os.environ.get("LLAMA_SERVER_URL", "").strip():
            return current(config, request)

        with autotune._AUTOTUNE_LOCK:
            managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
            managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
            if managed_process is not None and managed_process.poll() is None and managed_url:
                return managed_url

            binary = autotune._server_binary()
            if binary is None:
                raise RuntimeError("native llama-server binary is unavailable")
            model_path = autotune._resolve_model_path(config)
            batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)
            ubatch = min(batch, autotune._env_int("MMM_LLAMA_UBATCH", 512))
            width = _width(autotune)
            selected = autotune.ServerVariant(
                name=f"qwen35-hot-mtp-{width}",
                spec_type="draft-mtp",
                draft_n_max=width,
                ubatch=ubatch,
                parallel=1,
                cache_reuse=0,
                draft_p_min=0.0,
            )
            url = autotune._launch_selected(binary, model_path, config, selected)
            os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = "draft-mtp"
            os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = str(width)
            os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"
            return url

    setattr(ensure_qwen35_mtp3, _MARKER, True)
    autotune.ensure_tuned_server = ensure_qwen35_mtp3


__all__ = ["_is_qwen35_mtp", "install"]
