from __future__ import annotations

import json
import os
import signal
import time
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_DEFAULT_WIDTH = 3
_MARKER = "_mmm_qwen35_mtp3_hotpath"
_ACTIVE_RUNTIME_KEYS = (
    "MMM_LLAMA_ACTIVE_SPEC_TYPE",
    "MMM_LLAMA_ACTIVE_DRAFT_N_MAX",
    "MMM_LLAMA_ACTIVE_PARALLEL",
    "MMM_LLAMA_ACTIVE_UBATCH",
    "MMM_LLAMA_ACTIVE_CACHE_REUSE",
    "MMM_LLAMA_ACTIVE_MTP_P_MIN",
    "MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE",
    "MMM_LLAMA_ACTIVE_KV_CACHE",
)


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
    return "qwen3.5-9b" in model_id and ("mtp" in model_id or "mtp" in filename)


def _width(autotune: Any) -> int:
    return autotune._env_int("MMM_QWEN35_MTP_WIDTH", _DEFAULT_WIDTH, minimum=1)


def _prior_mmm_loopback_port() -> int | None:
    """Return the port of a server left by an earlier MMM Colab engine load."""

    receipt_raw = os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip()
    endpoint = os.environ.get("LLAMA_SERVER_URL", "").strip()
    if not receipt_raw or not endpoint:
        return None
    try:
        receipt = json.loads(receipt_raw)
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get("backend") != "local_cuda":
        return None
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or port is None:
        return None
    return int(port)


def _process_is_llama_server_on_port(pid: int, port: int) -> bool:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False
    args = [
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    ]
    if not args or Path(args[0]).name != "llama-server":
        return False
    target = str(port)
    for index, value in enumerate(args):
        if value.startswith("--port=") and value.partition("=")[2] == target:
            return True
        if value in {"--port", "-p"} and index + 1 < len(args):
            if args[index + 1] == target:
                return True
    return False


def _reclaim_prior_mmm_server() -> None:
    """Stop only the loopback llama-server recorded by a prior MMM Colab setup.

    Reloading GitHub main clears Python modules but cannot by itself terminate the
    native child process created by the old module. Without this guard, the refreshed
    engine can reconnect to that stale server and silently keep the old decode policy.
    """

    port = _prior_mmm_loopback_port()
    if port is None:
        return

    proc_root = Path("/proc")
    matched: list[int] = []
    try:
        pids = [int(path.name) for path in proc_root.iterdir() if path.name.isdigit()]
    except OSError:
        pids = []
    for pid in pids:
        if pid == os.getpid() or not _process_is_llama_server_on_port(pid, port):
            continue
        matched.append(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue

    if matched:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if all(not (proc_root / str(pid)).exists() for pid in matched):
                break
            time.sleep(0.05)
        for pid in matched:
            if not (proc_root / str(pid)).exists():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # The receipt proves this loopback URL belonged to the prior MMM local setup.
    # Clear all exported selection state so the refreshed engine cannot reuse it.
    os.environ.pop("LLAMA_SERVER_URL", None)
    for name in _ACTIVE_RUNTIME_KEYS:
        os.environ.pop(name, None)


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

        managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if managed_process is not None and managed_process.poll() is None and managed_url:
            return managed_url

        # A GitHub-main refresh can leave the old native child alive after its Python
        # module was discarded. Reclaim that exact prior MMM loopback server before
        # honoring any external-server shortcut in the generic layer.
        _reclaim_prior_mmm_server()
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


__all__ = ["_is_qwen35_mtp", "_prior_mmm_loopback_port", "install"]
