from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from functools import wraps
from pathlib import Path
from typing import Any

import httpx

from .llama_server_autotune import _resolve_model_path


LLAMA_CPP_PYTHON_VERSION = "0.3.34"
SERVER_SOURCE_URL = (
    "https://raw.githubusercontent.com/abetlen/llama-cpp-python/"
    f"v{LLAMA_CPP_PYTHON_VERSION}/examples/server/server.py"
)
SERVER_SOURCE_GIT_BLOB_SHA1 = "72adc790598eac9574aec6fc0bf6e994a9cfe732"
SERVER_SCRIPT_PATH = Path("/content/mmm_llama_mtp_server_v0_3_34.py")
SERVER_CONFIG_PATH = Path("/content/mmm_llama_mtp_server_v0_3_34.json")
SERVER_LOG_PATH = Path("/content/mmm_llama_mtp_server.log")
SERVER_ORIGIN = "http://127.0.0.1:8910"
SERVER_API_URL = f"{SERVER_ORIGIN}/v1"
SERVER_CONTEXT_CAP = 16384
ENABLED_ENV = "MMM_COLAB_MTP_SERVER_ENABLED"
START_TIMEOUT_ENV = "MMM_COLAB_MTP_SERVER_START_TIMEOUT"

_PROCESS: subprocess.Popen[str] | None = None
_LOG_HANDLE: Any = None


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _server_source() -> Path:
    if SERVER_SCRIPT_PATH.is_file():
        data = SERVER_SCRIPT_PATH.read_bytes()
        if _git_blob_sha1(data) == SERVER_SOURCE_GIT_BLOB_SHA1:
            return SERVER_SCRIPT_PATH

    request = urllib.request.Request(
        SERVER_SOURCE_URL,
        headers={"User-Agent": "M.M.M-Colab-MTP/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    actual = _git_blob_sha1(data)
    if actual != SERVER_SOURCE_GIT_BLOB_SHA1:
        raise RuntimeError(
            "Pinned MTP server source hash mismatch: "
            f"expected {SERVER_SOURCE_GIT_BLOB_SHA1}, got {actual}."
        )
    SERVER_SCRIPT_PATH.write_bytes(data)
    return SERVER_SCRIPT_PATH


def _ready() -> bool:
    try:
        response = httpx.get(f"{SERVER_API_URL}/models", timeout=1.0)
        if response.status_code != 200:
            return False
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return False
        return any(
            isinstance(model, dict) and str(model.get("id", "")) == "local"
            for model in models
        )
    except Exception:
        return False


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _close_log() -> None:
    global _LOG_HANDLE
    if _LOG_HANDLE is None:
        return
    try:
        _LOG_HANDLE.close()
    except Exception:
        pass
    _LOG_HANDLE = None


def _log_tail(lines: int = 80) -> str:
    try:
        values = SERVER_LOG_PATH.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except Exception:
        return ""
    return "\n".join(values[-lines:])


def _mtp_width() -> int:
    raw = os.environ.get("MMM_LLAMA_MTP_WIDTH", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 3
    return min(8, max(1, value))


def _start_timeout() -> int:
    raw = os.environ.get(START_TIMEOUT_ENV, "300").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 300
    return min(900, max(30, value))


def _cuda_backend_files(package_dir: Path) -> list[Path]:
    values: set[Path] = set()
    for pattern in ("libggml-cuda.so*", "ggml-cuda.dll", "libggml-cuda*.dylib"):
        values.update(path.resolve() for path in package_dir.rglob(pattern) if path.is_file())
    return sorted(values)


def _validate_cuda_binding() -> None:
    try:
        import llama_cpp
    except Exception as exc:
        raise RuntimeError("llama-cpp-python is not installed.") from exc

    version = str(getattr(llama_cpp, "__version__", ""))
    if version and version != LLAMA_CPP_PYTHON_VERSION:
        raise RuntimeError(
            "llama-cpp-python version mismatch: "
            f"expected {LLAMA_CPP_PYTHON_VERSION}, found {version}."
        )

    package_file = getattr(llama_cpp, "__file__", "") or ""
    if not package_file:
        raise RuntimeError("llama-cpp-python package path is unavailable.")
    backends = _cuda_backend_files(Path(package_file).resolve().parent)
    if not backends:
        raise RuntimeError(
            "llama-cpp-python CUDA backend library is missing from the installed wheel."
        )


def _write_config(config: Any, model_path: str) -> int:
    threads = max(1, min(8, os.cpu_count() or 1))
    width = _mtp_width()
    payload = {
        "server": {
            "host": "127.0.0.1",
            "port": 8910,
        },
        "model": {
            "path": model_path,
            "alias": "local",
            "n_ctx": min(int(config.max_context), SERVER_CONTEXT_CAP),
            "max_output_tokens": int(config.max_new_tokens),
            "n_seq_max": 1,
            "n_batch": 512,
            "n_ubatch": 512,
            "threads": threads,
            "threads_batch": threads,
            "kv_unified": True,
            "use_mmap": True,
            "use_mlock": False,
            "n_gpu_layers": -1,
            "flash_attn": True,
            "offload_kqv": True,
            "draft_model": "draft-mtp",
            "draft_model_num_pred_tokens": width,
            "draft_model_threads": max(1, min(4, threads)),
            "draft_model_threads_batch": threads,
        },
    }
    SERVER_CONFIG_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return width


def colab_mtp_server_enabled() -> bool:
    raw = os.environ.get(ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def colab_mtp_server_running() -> bool:
    return _PROCESS is not None and _PROCESS.poll() is None and _ready()


def stop_colab_mtp_server(*, keep_enabled: bool = True) -> None:
    """Release the MTP server process and its GPU allocation."""

    global _PROCESS
    _stop_process(_PROCESS)
    _PROCESS = None
    _close_log()
    if os.environ.get("LLAMA_SERVER_URL", "").rstrip("/") == SERVER_API_URL:
        os.environ.pop("LLAMA_SERVER_URL", None)
    if not keep_enabled:
        os.environ.pop(ENABLED_ENV, None)


def start_colab_mtp_server(config: Any) -> str:
    """Start the pinned CUDA MTP server and return its OpenAI-compatible URL."""

    global _PROCESS, _LOG_HANDLE

    os.environ[ENABLED_ENV] = "1"

    if _PROCESS is not None and _PROCESS.poll() is None:
        if _ready():
            os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
            print("MTP server: ready", SERVER_API_URL, flush=True)
            return SERVER_API_URL
        stop_colab_mtp_server(keep_enabled=True)
    elif _ready():
        raise RuntimeError(
            "Port 8910 is already serving a different unmanaged process. "
            "Stop that process before starting the M.M.M MTP server."
        )

    print("MTP server: checking CUDA binding", flush=True)
    _validate_cuda_binding()

    print("MTP server: resolving model", flush=True)
    model_path = _resolve_model_path(config)
    print("MTP server: model ready", Path(model_path).name, flush=True)

    print("MTP server: preparing launcher", flush=True)
    server_script = _server_source()
    width = _write_config(config, model_path)

    _LOG_HANDLE = SERVER_LOG_PATH.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(server_script),
        "-C",
        str(SERVER_CONFIG_PATH),
    ]
    print("MTP server: starting", f"draft_width={width}", flush=True)
    _PROCESS = subprocess.Popen(
        command,
        stdout=_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    started = time.monotonic()
    deadline = started + _start_timeout()
    next_status = started + 10
    while time.monotonic() < deadline:
        returncode = _PROCESS.poll()
        if returncode is not None:
            try:
                _LOG_HANDLE.flush()
            except Exception:
                pass
            tail = _log_tail()
            _PROCESS = None
            _close_log()
            raise RuntimeError(
                f"MTP server exited during startup with code {returncode}."
                + ("\n" + tail if tail else "")
            )
        if _ready():
            os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
            elapsed = time.monotonic() - started
            print(
                "MTP server: ready",
                SERVER_API_URL,
                f"startup={elapsed:.1f}s",
                flush=True,
            )
            return SERVER_API_URL
        now = time.monotonic()
        if now >= next_status:
            print(
                "MTP server: starting",
                f"elapsed={now - started:.0f}s",
                flush=True,
            )
            next_status = now + 10
        time.sleep(0.5)

    _stop_process(_PROCESS)
    _PROCESS = None
    if _LOG_HANDLE is not None:
        try:
            _LOG_HANDLE.flush()
        except Exception:
            pass
    tail = _log_tail()
    _close_log()
    raise RuntimeError(
        f"MTP server startup timed out after {_start_timeout()} seconds."
        + ("\n" + tail if tail else "")
    )


def _install_planner_request_budget() -> None:
    """Make request paging respect the actual managed-server context window."""

    from . import game_design as game_design_module

    original = game_design_module._request_page_bytes
    if getattr(original, "_mmm_mtp_context_budget", False):
        return

    @wraps(original)
    def request_page_bytes(router: Any = None, role: str = "planner") -> int:
        baseline = int(original(router, role))
        if router is None or not colab_mtp_server_enabled():
            return baseline
        try:
            config = router.registry.role(router.profile, role)
            configured_context = int(getattr(config, "max_context", 0) or 0)
            if configured_context <= 0:
                return baseline
            context = min(configured_context, SERVER_CONTEXT_CAP)
            max_output = max(0, int(getattr(config, "max_new_tokens", 0) or 0))
            # The server context is prompt + completion. Reserve the configured
            # completion budget and fixed chat/schema overhead instead of pretending
            # the registry's larger native model context is the active server context.
            available_tokens = max(1024, context - max_output - 2048)
            server_page_budget = max(
                4 * 1024,
                min(64 * 1024, int(available_tokens * 3.5)),
            )
            return min(baseline, server_page_budget)
        except Exception:
            return baseline

    request_page_bytes._mmm_mtp_context_budget = True
    game_design_module._request_page_bytes = request_page_bytes


_install_planner_request_budget()
atexit.register(lambda: stop_colab_mtp_server(keep_enabled=False))
