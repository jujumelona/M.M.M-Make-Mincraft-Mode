from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
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
            "Pinned llama-cpp-python MTP server source hash mismatch: "
            f"expected {SERVER_SOURCE_GIT_BLOB_SHA1}, got {actual}."
        )
    SERVER_SCRIPT_PATH.write_bytes(data)
    return SERVER_SCRIPT_PATH


def _ready() -> bool:
    try:
        response = httpx.get(f"{SERVER_API_URL}/models", timeout=1.0)
        return response.status_code == 200
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


def _log_tail(lines: int = 60) -> str:
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
            "n_ctx": min(int(config.max_context), 16384),
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


def start_colab_mtp_server(config: Any) -> str:
    """Start the pinned low-level MTP server using the installed CUDA wheel.

    This path downloads only the pinned Python server script. It never clones or
    compiles llama.cpp source. Failure is surfaced with the server log tail.
    """

    global _PROCESS, _LOG_HANDLE

    if _ready():
        os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
        print("llama MTP server: ready", SERVER_API_URL, flush=True)
        return SERVER_API_URL

    _stop_process(_PROCESS)
    _PROCESS = None
    if _LOG_HANDLE is not None:
        try:
            _LOG_HANDLE.close()
        except Exception:
            pass
        _LOG_HANDLE = None

    server_script = _server_source()
    model_path = _resolve_model_path(config)
    width = _write_config(config, model_path)

    _LOG_HANDLE = SERVER_LOG_PATH.open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(server_script),
        "-C",
        str(SERVER_CONFIG_PATH),
    ]
    print("llama MTP server: starting", f"draft_width={width}", flush=True)
    _PROCESS = subprocess.Popen(
        command,
        stdout=_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.monotonic() + 300
    next_status = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _PROCESS.poll() is not None:
            break
        if _ready():
            os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
            print("llama MTP server: ready", SERVER_API_URL, flush=True)
            return SERVER_API_URL
        now = time.monotonic()
        if now >= next_status:
            print("llama MTP server: model loading", flush=True)
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
    raise RuntimeError(
        "llama MTP server failed to become ready."
        + ("\n" + tail if tail else "")
    )
