from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
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
MTP_POLICY_ENV = "MMM_LLAMA_MTP_POLICY"
MTP_PROBE_TIMEOUT_ENV = "MMM_LLAMA_MTP_PROBE_TIMEOUT"
KV_CACHE_QUANT_ENV = "MMM_KV_CACHE_QUANT"
KV_CACHE_QUANTS = frozenset({"q4_0", "q8_0", "f16"})

_PROCESS: subprocess.Popen[str] | None = None
_LOG_HANDLE: Any = None
_SERVER_MODE: str | None = None
_MTP_DISABLED_REASON: str | None = None
_MTP_VERIFIED_KEYS: set[tuple[str, str, int]] = set()


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


def server_log_tail(lines: int = 80) -> str:
    try:
        values = SERVER_LOG_PATH.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except Exception:
        return ""
    return "\n".join(values[-lines:])


def server_log_offset() -> int:
    try:
        return SERVER_LOG_PATH.stat().st_size
    except OSError:
        return 0


def server_log_since(offset: int, *, max_bytes: int = 64 * 1024) -> str:
    try:
        size = SERVER_LOG_PATH.stat().st_size
        start = min(max(0, int(offset)), size)
        if size - start > max_bytes:
            start = size - max_bytes
        with SERVER_LOG_PATH.open("rb") as handle:
            handle.seek(start)
            data = handle.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def decode_log_summary(text: str) -> str:
    values = [
        int(match)
        for match in re.findall(r"\bn_decoded\s*=\s*(\d+)", text or "")
    ]
    if not values:
        return "no n_decoded telemetry in request log"
    first = values[0]
    last = values[-1]
    maximum = max(values)
    if maximum <= 1:
        state = "decode stalled at or before the first decoded token"
    elif last >= 2:
        state = "server-side decode advanced beyond the first token"
    else:
        state = "server-side decode telemetry is non-monotonic"
    return (
        f"{state}; n_decoded first={first} last={last} "
        f"max={maximum} samples={len(values)}"
    )


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


def _mtp_probe_timeout() -> float:
    raw = os.environ.get(MTP_PROBE_TIMEOUT_ENV, "45").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 45.0
    return min(180.0, max(10.0, value))


def _mtp_policy() -> str:
    value = os.environ.get(MTP_POLICY_ENV, "auto").strip().lower()
    if value not in {"auto", "off"}:
        return "auto"
    return value


def _kv_cache_quant() -> str:
    value = os.environ.get(KV_CACHE_QUANT_ENV, "q4_0").strip().lower()
    if value not in KV_CACHE_QUANTS:
        raise RuntimeError(
            f"{KV_CACHE_QUANT_ENV} must be one of {sorted(KV_CACHE_QUANTS)}, got {value!r}."
        )
    return value


def _kv_cache_type_id() -> int:
    try:
        import llama_cpp.llama_cpp as ggml
    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python constants are unavailable for KV cache configuration."
        ) from exc
    mapping = {
        "q4_0": int(ggml.GGML_TYPE_Q4_0),
        "q8_0": int(ggml.GGML_TYPE_Q8_0),
        "f16": int(ggml.GGML_TYPE_F16),
    }
    return mapping[_kv_cache_quant()]


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


def _mtp_capable(config: Any) -> bool:
    if _mtp_policy() == "off" or _MTP_DISABLED_REASON:
        return False
    extra = getattr(config, "extra", {})
    if isinstance(extra, dict) and "mtp_capable" in extra:
        return bool(extra["mtp_capable"])
    labels = [
        str(getattr(config, "model_id", "")),
        str(extra.get("gguf_filename", "")) if isinstance(extra, dict) else "",
    ]
    return any("MTP" in value.upper() for value in labels if value)


def _structured_response_format(request: Any | None) -> bool:
    if request is None:
        return False
    value = getattr(request, "response_format", None)
    if isinstance(value, str):
        return value.strip().lower() in {
            "json",
            "json_object",
            "json_schema",
            "structured",
        }
    if isinstance(value, dict):
        kind = str(value.get("type", "")).strip().lower()
        return kind in {"json_object", "json_schema"}
    return False


def request_server_mode(config: Any, request: Any | None = None) -> str:
    """Choose the narrowest safe server mode for this request.

    The pinned server has no request-level switch for speculative decoding. Structured
    planner/coder/repair pages therefore run on the baseline target context and are
    host-validated instead of combining MTP with a server-side JSON grammar. MTP is
    reserved for unconstrained text generation and only after a live multi-step
    streaming probe succeeds for this model/width.
    """

    if _structured_response_format(request):
        return "baseline"
    return "mtp" if _mtp_capable(config) else "baseline"


def current_server_mode() -> str | None:
    return _SERVER_MODE if colab_mtp_server_running() else None


def _write_config(config: Any, model_path: str, *, mode: str) -> int:
    if mode not in {"baseline", "mtp"}:
        raise ValueError(f"Unknown Colab llama server mode: {mode}")
    threads = max(1, min(8, os.cpu_count() or 1))
    width = _mtp_width() if mode == "mtp" else 0
    kv_type = _kv_cache_type_id()
    model_config: dict[str, Any] = {
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
        "type_k": kv_type,
        "type_v": kv_type,
    }
    if mode == "mtp":
        model_config.update(
            {
                "draft_model": "draft-mtp",
                "draft_model_num_pred_tokens": width,
                "draft_model_threads": max(1, min(4, threads)),
                "draft_model_threads_batch": threads,
            }
        )
    payload = {
        "server": {"host": "127.0.0.1", "port": 8910},
        "model": model_config,
    }
    SERVER_CONFIG_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return width


def colab_mtp_server_enabled() -> bool:
    raw = os.environ.get(ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def colab_mtp_server_running(*, mode: str | None = None) -> bool:
    running = _PROCESS is not None and _PROCESS.poll() is None and _ready()
    if not running:
        return False
    return mode is None or _SERVER_MODE == mode


def stop_colab_mtp_server(*, keep_enabled: bool = True) -> None:
    """Release the managed llama server process and its GPU allocation."""

    global _PROCESS, _SERVER_MODE
    _stop_process(_PROCESS)
    _PROCESS = None
    _SERVER_MODE = None
    _close_log()
    if os.environ.get("LLAMA_SERVER_URL", "").rstrip("/") == SERVER_API_URL:
        os.environ.pop("LLAMA_SERVER_URL", None)
    if not keep_enabled:
        os.environ.pop(ENABLED_ENV, None)


def start_colab_mtp_server(config: Any, *, mode: str = "baseline") -> str:
    """Start the pinned CUDA server in a request-safe baseline or MTP mode."""

    global _PROCESS, _LOG_HANDLE, _SERVER_MODE

    if mode == "mtp" and not _mtp_capable(config):
        mode = "baseline"
    if mode not in {"baseline", "mtp"}:
        raise ValueError(f"Unknown Colab llama server mode: {mode}")

    os.environ[ENABLED_ENV] = "1"

    if _PROCESS is not None and _PROCESS.poll() is None:
        if _ready() and _SERVER_MODE == mode:
            os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
            return SERVER_API_URL
        stop_colab_mtp_server(keep_enabled=True)
    elif _ready():
        raise RuntimeError(
            "Port 8910 is already serving a different unmanaged process. "
            "Stop that process before starting the M.M.M llama server."
        )

    print("llama server: checking CUDA binding", flush=True)
    _validate_cuda_binding()

    print("llama server: resolving model", flush=True)
    model_path = _resolve_model_path(config)
    print("llama server: model ready", Path(model_path).name, flush=True)

    print(
        "llama server: preparing launcher",
        f"mode={mode}",
        f"kv={_kv_cache_quant()}",
        flush=True,
    )
    server_script = _server_source()
    width = _write_config(config, model_path, mode=mode)

    _LOG_HANDLE = SERVER_LOG_PATH.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(server_script),
        "-C",
        str(SERVER_CONFIG_PATH),
    ]
    mode_detail = f"draft_width={width}" if mode == "mtp" else "speculation=off"
    print(
        "llama server: starting",
        f"mode={mode}",
        mode_detail,
        f"kv={_kv_cache_quant()}",
        flush=True,
    )
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
            tail = server_log_tail()
            _PROCESS = None
            _SERVER_MODE = None
            _close_log()
            raise RuntimeError(
                f"llama server exited during startup with code {returncode}."
                + ("\n" + tail if tail else "")
            )
        if _ready():
            _SERVER_MODE = mode
            os.environ["LLAMA_SERVER_URL"] = SERVER_API_URL
            elapsed = time.monotonic() - started
            print(
                "llama server: ready",
                SERVER_API_URL,
                f"mode={mode}",
                f"kv={_kv_cache_quant()}",
                f"startup={elapsed:.1f}s",
                flush=True,
            )
            return SERVER_API_URL
        now = time.monotonic()
        if now >= next_status:
            print(
                "llama server: starting",
                f"mode={mode}",
                f"elapsed={now - started:.0f}s",
                flush=True,
            )
            next_status = now + 10
        time.sleep(0.5)

    _stop_process(_PROCESS)
    _PROCESS = None
    _SERVER_MODE = None
    if _LOG_HANDLE is not None:
        try:
            _LOG_HANDLE.flush()
        except Exception:
            pass
    tail = server_log_tail()
    _close_log()
    raise RuntimeError(
        f"llama server startup timed out after {_start_timeout()} seconds."
        + ("\n" + tail if tail else "")
    )


def _mtp_probe_key(config: Any) -> tuple[str, str, int]:
    extra = getattr(config, "extra", {})
    filename = str(extra.get("gguf_filename", "")) if isinstance(extra, dict) else ""
    return str(getattr(config, "model_id", "")), filename, _mtp_width()


def _probe_mtp_server() -> tuple[bool, str]:
    """Exercise actual MTP decode *and* SSE delivery across several decode steps."""

    payload = {
        "model": "local",
        "messages": [
            {
                "role": "system",
                "content": "Follow the user literally. Do not explain.",
            },
            {
                "role": "user",
                "content": (
                    "Output the integers 1 through 20 in order, separated by one "
                    "space, and nothing else."
                ),
            },
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "stream": True,
    }
    started = time.monotonic()
    content_chars = 0
    output_events = 0
    saw_done = False
    try:
        timeout = _mtp_probe_timeout()
        with httpx.stream(
            "POST",
            f"{SERVER_API_URL}/chat/completions",
            json=payload,
            timeout=httpx.Timeout(
                connect=min(30.0, timeout),
                read=timeout,
                write=min(30.0, timeout),
                pool=min(30.0, timeout),
            ),
        ) as response:
            if response.status_code != 200:
                response.read()
                return False, f"HTTP {response.status_code}"
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                parts = (
                    delta.get("reasoning_content"),
                    delta.get("reasoning"),
                    delta.get("content"),
                )
                emitted = "".join(part for part in parts if isinstance(part, str))
                if emitted:
                    output_events += 1
                    content_chars += len(emitted)
        elapsed = time.monotonic() - started
        if not saw_done:
            return False, f"stream ended before [DONE] after {elapsed:.1f}s"
        if content_chars < 8:
            return False, (
                f"insufficient streamed output chars={content_chars} "
                f"events={output_events} elapsed={elapsed:.1f}s"
            )
        return True, (
            f"stream_chars={content_chars} events={output_events} "
            f"elapsed={elapsed:.1f}s"
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        return False, f"{type(exc).__name__}: {exc} after {elapsed:.1f}s"


def mark_mtp_unhealthy(reason: str) -> None:
    global _MTP_DISABLED_REASON
    rendered = reason.strip() or "runtime MTP failure"
    if _MTP_DISABLED_REASON is None:
        _MTP_DISABLED_REASON = rendered
        print("llama server: MTP disabled for this runtime", rendered, flush=True)


def ensure_colab_server_for_request(config: Any, request: Any) -> str:
    """Ensure structured calls use baseline and only verified text calls use MTP."""

    desired = request_server_mode(config, request)
    url = start_colab_mtp_server(config, mode=desired)
    if desired != "mtp":
        return url

    key = _mtp_probe_key(config)
    if key in _MTP_VERIFIED_KEYS:
        return url

    print("llama server: verifying MTP text decode", f"width={key[2]}", flush=True)
    ok, detail = _probe_mtp_server()
    if ok:
        _MTP_VERIFIED_KEYS.add(key)
        print("llama server: MTP probe PASS", detail, flush=True)
        return url

    mark_mtp_unhealthy("probe failed: " + detail)
    stop_colab_mtp_server(keep_enabled=True)
    return start_colab_mtp_server(config, mode="baseline")


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
