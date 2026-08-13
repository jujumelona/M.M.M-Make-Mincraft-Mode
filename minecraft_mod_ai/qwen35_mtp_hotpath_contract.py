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
_DEFAULT_CTX = 8192
_MIN_CTX = 4096
_MAX_CTX = 2147483647
_MARKER = "_mmm_qwen35_mtp3_hotpath_v5"
_BASE_MARKER = "_mmm_qwen35_measured_fast_args_v4"
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


def _context_size(config: Any | None = None) -> int:
    """Return the bounded production KV window for the Qwen3.5 T4 hot path.

    MMM pages planning and source evidence, so the default server should not reserve
    the model's entire context window. Operators may explicitly raise the window for
    an exceptional request without enabling any runtime search sweep.
    """

    configured_max = getattr(config, "max_context", _MAX_CTX)
    try:
        upper = int(configured_max)
    except (TypeError, ValueError):
        upper = _MAX_CTX
    upper = max(_MIN_CTX, min(_MAX_CTX, upper))

    raw = os.environ.get("MMM_QWEN35_MTP_CTX", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_CTX
    except ValueError:
        value = _DEFAULT_CTX
    return max(_MIN_CTX, min(upper, value))


def _drop_option(
    args: list[str],
    names: tuple[str, ...],
    *,
    takes_value: bool = True,
) -> None:
    for name in names:
        while name in args:
            index = args.index(name)
            del args[index]
            if takes_value and index < len(args):
                del args[index]


def _set_option(args: list[str], names: tuple[str, ...], value: str) -> None:
    for name in names:
        if name in args:
            index = args.index(name)
            if index + 1 < len(args):
                args[index + 1] = value
                return
    args.extend([names[0], value])


def _install_measured_fast_base_args(autotune: Any) -> None:
    """Install the measured single-stream Qwen3.5 server launch shape."""

    current = getattr(autotune, "_base_args", None)
    if not callable(current) or getattr(current, _BASE_MARKER, False):
        return

    @wraps(current)
    def measured_base_args(
        binary: str,
        model_path: str,
        config: Any,
        port: int,
    ) -> list[str]:
        args = list(current(binary, model_path, config, port))
        if not _enabled() or not _is_qwen35_mtp(config):
            return args

        _set_option(args, ("--gpu-layers", "-ngl"), "all")
        _set_option(args, ("--flash-attn", "-fa"), "on")
        _set_option(args, ("--batch-size", "-b"), "2048")
        _set_option(args, ("--ubatch-size", "-ub"), "512")
        _set_option(args, ("--ctx-size", "-c"), str(_context_size(config)))

        # Preserve llama.cpp's native KV format. Runtime KV sweeps repeatedly reload
        # a large model and belong in offline benchmarking, not production requests.
        _drop_option(args, ("--cache-type-k", "-ctk"))
        _drop_option(args, ("--cache-type-v", "-ctv"))
        _drop_option(args, ("--load-mode", "-lm"))
        _drop_option(args, ("--cache-prompt",), takes_value=False)
        if "--metrics" not in args:
            args.append("--metrics")
        return args

    setattr(measured_base_args, _BASE_MARKER, True)
    autotune._base_args = measured_base_args


def _prior_mmm_loopback_port() -> int | None:
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
    """Stop only the loopback llama-server recorded by a prior MMM Colab setup."""

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

    os.environ.pop("LLAMA_SERVER_URL", None)
    for name in _ACTIVE_RUNTIME_KEYS:
        os.environ.pop(name, None)


def install(autotune: Any) -> None:
    """Install one fixed Qwen3.5 MTP-3 single-stream production server."""

    _install_measured_fast_base_args(autotune)
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

        _reclaim_prior_mmm_server()
        if os.environ.get("LLAMA_SERVER_URL", "").strip():
            return current(config, request)

        with autotune._AUTOTUNE_LOCK:
            managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
            managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
            if (
                managed_process is not None
                and managed_process.poll() is None
                and managed_url
            ):
                return managed_url

            binary = autotune._server_binary()
            if binary is None:
                raise RuntimeError("native llama-server binary is unavailable")
            model_path = autotune._resolve_model_path(config)
            selected = autotune.ServerVariant(
                name="qwen35-production-mtp-3",
                spec_type="draft-mtp",
                draft_n_max=_DEFAULT_WIDTH,
                ubatch=512,
                parallel=1,
                cache_reuse=0,
                draft_p_min=0.0,
            )
            url = autotune._launch_selected(binary, model_path, config, selected)
            os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = "draft-mtp"
            os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = str(_DEFAULT_WIDTH)
            os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"
            os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = "512"
            os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = "0"
            os.environ["MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE"] = "single_stream"
            os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = "native-default"
            print(
                "llama server: Qwen3.5 fixed production profile",
                (
                    "spec=draft-mtp n_max=3 parallel=1 "
                    f"ctx={_context_size(config)} ubatch=512 kv=native-default"
                ),
                flush=True,
            )
            return url

    setattr(ensure_qwen35_mtp3, _MARKER, True)
    ensure_qwen35_mtp3._mmm_qwen35_mtp3_hotpath = True  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_qwen35_mtp3


__all__ = [
    "_context_size",
    "_install_measured_fast_base_args",
    "_is_qwen35_mtp",
    "_prior_mmm_loopback_port",
    "install",
]