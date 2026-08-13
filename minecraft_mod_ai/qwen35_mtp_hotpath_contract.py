from __future__ import annotations

import json
import os
import signal
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit


_DEFAULT_WIDTH = 3
_MARKER = "_mmm_qwen35_adaptive_mtp_hotpath_v2"
_BASE_MARKER = "_mmm_qwen35_measured_fast_args"
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
_QWEN_SEARCH_DEFAULTS = {
    # Qwen3.5 MTP has a native draft head. Compare the old width-3 path against
    # nearby widths and confidence-gated wider drafts instead of assuming one
    # historical benchmark remains optimal on every CUDA/runtime combination.
    "MMM_LLAMA_MTP_WIDTHS": "2,3",
    "MMM_LLAMA_MTP_CONFIDENCE_WIDTHS": "3,6,8",
    "MMM_LLAMA_MTP_SEED_P_MIN": "0.8",
    "MMM_LLAMA_MTP_P_MIN_CANDIDATES": "0,0.6,0.8,0.9",
    # Native MTP is the relevant speculative family for this checkpoint. N-gram
    # candidates only add repeated 6 GB model reloads to first-run tuning.
    "MMM_LLAMA_NGRAM_SPEC_TYPES": "",
    # The Qwen measured base args intentionally use llama.cpp's native KV layout.
    # The generic KV sweep would benchmark q4/q8/f16 labels after those CLI flags
    # have already been removed, so it would reload the same effective server.
    "MMM_LLAMA_KV_AUTOTUNE": "0",
    "MMM_LLAMA_TUNING_OBJECTIVE": "single_stream",
}


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
    """Compatibility helper for callers that explicitly pin a historical width."""

    return autotune._env_int("MMM_QWEN35_MTP_WIDTH", _DEFAULT_WIDTH, minimum=1)


def _drop_option(args: list[str], names: tuple[str, ...], *, takes_value: bool = True) -> None:
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
    """Keep the server-side Qwen3.5 launch shape that previously measured fastest.

    Full GPU offload, FlashAttention, batch 2048 and a single decode slot remain
    fixed. Speculative width/confidence are deliberately *not* fixed here: the
    adaptive decode benchmark chooses them from byte-identical candidates.
    """

    current = getattr(autotune, "_base_args", None)
    if not callable(current) or getattr(current, _BASE_MARKER, False):
        return

    @wraps(current)
    def measured_base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
        args = list(current(binary, model_path, config, port))
        if not _enabled() or not _is_qwen35_mtp(config):
            return args

        _set_option(args, ("--gpu-layers", "-ngl"), "all")
        _set_option(args, ("--flash-attn", "-fa"), "on")
        _set_option(args, ("--batch-size", "-b"), "2048")
        _set_option(args, ("--ubatch-size", "-ub"), "512")

        # Preserve the actual historical fast launch rather than inheriting generic
        # cache experiments. Context is intentionally untouched for correctness.
        _drop_option(args, ("--cache-type-k", "-ctk"))
        _drop_option(args, ("--cache-type-v", "-ctv"))
        _drop_option(args, ("--load-mode", "-lm"))
        _drop_option(args, ("--cache-prompt",), takes_value=False)
        if "--metrics" not in args:
            args.append("--metrics")
        return args

    setattr(measured_base_args, _BASE_MARKER, True)
    autotune._base_args = measured_base_args


@contextmanager
def _qwen_speed_search_defaults() -> Iterator[None]:
    """Apply focused first-run search defaults without overriding user tuning."""

    inserted: list[str] = []
    for name, value in _QWEN_SEARCH_DEFAULTS.items():
        if os.environ.get(name, "").strip():
            continue
        os.environ[name] = value
        inserted.append(name)
    try:
        yield
    finally:
        for name in inserted:
            os.environ.pop(name, None)


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


def _active_runtime_summary() -> str:
    spec = os.environ.get("MMM_LLAMA_ACTIVE_SPEC_TYPE", "none") or "none"
    width = os.environ.get("MMM_LLAMA_ACTIVE_DRAFT_N_MAX", "0") or "0"
    p_min = os.environ.get("MMM_LLAMA_ACTIVE_MTP_P_MIN", "0") or "0"
    ubatch = os.environ.get("MMM_LLAMA_ACTIVE_UBATCH", "")
    fields = [f"spec={spec}", f"n_max={width}", f"p_min={p_min}"]
    if ubatch:
        fields.append(f"ubatch={ubatch}")
    return " ".join(fields)


def install(autotune: Any) -> None:
    """Autotune Qwen3.5 MTP instead of pinning one historical speculative width."""

    _install_measured_fast_base_args(autotune)

    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_qwen35_fastest(config: Any, request: Any) -> str:
        if not _enabled() or not _is_qwen35_mtp(config):
            return current(config, request)

        managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if managed_process is not None and managed_process.poll() is None and managed_url:
            return managed_url

        # A stale MMM server may have been recorded by a previous notebook engine
        # import. Reclaim only that exact loopback process before benchmarking.
        _reclaim_prior_mmm_server()

        with _qwen_speed_search_defaults():
            url = current(config, request)

        # The Qwen base profile deliberately uses llama.cpp's native KV layout.
        os.environ.setdefault("MMM_LLAMA_ACTIVE_KV_CACHE", "native-default")
        print(
            "llama server: Qwen3.5 decode profile selected",
            _active_runtime_summary(),
            flush=True,
        )
        return url

    setattr(ensure_qwen35_fastest, _MARKER, True)
    # Preserve the old marker as a compatibility signal for older runtime guards,
    # while no longer implying that width 3 is forcibly selected.
    ensure_qwen35_fastest._mmm_qwen35_mtp3_hotpath = True  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_qwen35_fastest


__all__ = [
    "_install_measured_fast_base_args",
    "_is_qwen35_mtp",
    "_prior_mmm_loopback_port",
    "_qwen_speed_search_defaults",
    "install",
]
