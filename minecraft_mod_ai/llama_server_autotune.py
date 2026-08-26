from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_AUTOTUNE_LOCK = threading.RLock()
_MANAGED_PROCESS: subprocess.Popen[bytes] | None = None
_MANAGED_URL: str | None = None
_MANAGED_KEY: str | None = None
_ATTEMPTED_KEYS: set[str] = set()
_BENCHMARK_SCHEMA_VERSION = "mmm/llama-server-autotune-v2-compact"
_BENCHMARK_OUTPUT_TOKENS = 96


@dataclass(frozen=True)
class ServerVariant:
    name: str
    spec_type: str = "none"
    draft_n_max: int = 0


@dataclass(frozen=True)
class ProbeResult:
    variant: ServerVariant
    ok: bool
    output_sha256: str
    predicted_tokens: int
    predicted_tps: float
    prompt_tps: float
    elapsed_seconds: float
    error: str = ""


@dataclass(frozen=True)
class AutotuneDecision:
    fingerprint: str
    selected: ServerVariant
    baseline_tps: float
    selected_tps: float
    speedup: float
    probes: tuple[ProbeResult, ...]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _candidate_variants() -> tuple[ServerVariant, ...]:
    """Keep exhaustive width selection while avoiding work unrelated to decoding."""

    raw = os.environ.get("MMM_LLAMA_MTP_WIDTHS", "1,2,3")
    widths: list[int] = []
    for token in raw.split(","):
        try:
            width = int(token.strip())
        except ValueError:
            continue
        if 1 <= width <= 8 and width not in widths:
            widths.append(width)
    return (ServerVariant("baseline"),) + tuple(
        ServerVariant(f"mtp-{width}", "draft-mtp", width) for width in widths
    )


def _choose_variant(
    probes: Iterable[ProbeResult],
    *,
    minimum_speedup: float,
) -> AutotuneDecision | None:
    values = tuple(probes)
    baseline = next(
        (probe for probe in values if probe.variant.name == "baseline" and probe.ok),
        None,
    )
    if baseline is None or baseline.predicted_tps <= 0:
        return None

    # Greedy speculative decoding is eligible only when it commits byte-identical
    # output to the baseline on the same deterministic probe.
    eligible = [baseline]
    eligible.extend(
        probe
        for probe in values
        if probe.ok
        and probe.variant.name != "baseline"
        and probe.output_sha256 == baseline.output_sha256
        and probe.predicted_tps > 0
    )
    fastest = max(eligible, key=lambda probe: probe.predicted_tps)
    required = baseline.predicted_tps * max(1.0, minimum_speedup)
    selected = fastest if fastest.predicted_tps >= required else baseline
    return AutotuneDecision(
        fingerprint="",
        selected=selected.variant,
        baseline_tps=baseline.predicted_tps,
        selected_tps=selected.predicted_tps,
        speedup=selected.predicted_tps / baseline.predicted_tps,
        probes=values,
    )


def _server_binary() -> str | None:
    explicit = (
        os.environ.get("MMM_LLAMA_SERVER_BIN", "").strip()
        or os.environ.get("LLAMA_SERVER_BIN", "").strip()
    )
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() and path.is_file():
            return str(path.resolve())
        return None
    return shutil.which("llama-server")


def _server_version(binary: str) -> str:
    try:
        completed = subprocess.run(
            [binary, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
        return completed.stdout.decode("utf-8", errors="replace").strip()[:512]
    except Exception:
        return "unknown"


def _hardware_identity() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        value = completed.stdout.decode("utf-8", errors="replace").strip()
        if value:
            return value
    except Exception:
        pass
    return "unknown-gpu"


def _resolve_model_path_direct(config: Any) -> str:
    model_id = str(config.model_id)
    candidate = Path(model_id).expanduser()
    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".gguf":
        return str(candidate.resolve())

    from huggingface_hub import hf_hub_download

    repo_id = model_id
    if "/" not in repo_id:
        repo_id = f"bartowski/{repo_id}-GGUF"
    elif not repo_id.lower().endswith("-gguf") and "gguf" not in repo_id.lower():
        repo_id = f"bartowski/{repo_id.split('/')[-1]}-GGUF"

    filename = str(config.extra.get("gguf_filename", "")).strip()
    if not filename:
        from huggingface_hub import list_repo_files

        files = [name for name in list_repo_files(repo_id) if name.endswith(".gguf")]
        preferred = [
            name
            for name in files
            if "Q4_K_M" in name or "q4_k_m" in name or "Q4_0" in name
        ]
        if not files:
            raise RuntimeError(f"No GGUF file found in {repo_id!r}.")
        filename = (preferred or files)[0]
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _resolve_model_path(config: Any) -> str:
    from .parallel_runtime_contract import resolve_model_path

    return resolve_model_path(config, _resolve_model_path_direct)


def _fingerprint(config: Any, binary: str, model_path: str) -> str:
    path = Path(model_path)
    stat = path.stat()
    payload = {
        "schema": _BENCHMARK_SCHEMA_VERSION,
        "model_id": str(config.model_id),
        "gguf_filename": str(config.extra.get("gguf_filename", "")),
        "model_path": str(path.resolve()),
        "model_size": int(stat.st_size),
        "model_mtime_ns": int(stat.st_mtime_ns),
        "max_context": int(config.max_context),
        "kv": os.environ.get("MMM_KV_CACHE_QUANT", "q4_0").lower(),
        "server": _server_version(binary),
        "hardware": _hardware_identity(),
        "batch": _env_int("MMM_LLAMA_BATCH", 2048),
        "ubatch": _env_int("MMM_LLAMA_UBATCH", 512),
        "probe_tokens": _env_int(
            "MMM_LLAMA_AUTOTUNE_TOKENS", _BENCHMARK_OUTPUT_TOKENS
        ),
        "variants": [asdict(value) for value in _candidate_variants()],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_path() -> Path:
    raw = os.environ.get("MMM_LLAMA_AUTOTUNE_CACHE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".cache" / "mmm" / "llama-server-autotune.json").resolve()


def _load_cached_decision(fingerprint: str) -> AutotuneDecision | None:
    path = _cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("schema_version") != _BENCHMARK_SCHEMA_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    try:
        selected = ServerVariant(**payload["selected"])
        probes = tuple(
            ProbeResult(
                variant=ServerVariant(**item["variant"]),
                ok=bool(item["ok"]),
                output_sha256=str(item["output_sha256"]),
                predicted_tokens=int(item["predicted_tokens"]),
                predicted_tps=float(item["predicted_tps"]),
                prompt_tps=float(item["prompt_tps"]),
                elapsed_seconds=float(item["elapsed_seconds"]),
                error=str(item.get("error", "")),
            )
            for item in payload.get("probes", [])
        )
        return AutotuneDecision(
            fingerprint=fingerprint,
            selected=selected,
            baseline_tps=float(payload["baseline_tps"]),
            selected_tps=float(payload["selected_tps"]),
            speedup=float(payload["speedup"]),
            probes=probes,
        )
    except Exception:
        return None


def _save_decision(decision: AutotuneDecision) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "fingerprint": decision.fingerprint,
        "selected": asdict(decision.selected),
        "baseline_tps": decision.baseline_tps,
        "selected_tps": decision.selected_tps,
        "speedup": decision.speedup,
        "probes": [
            {
                **asdict(probe),
                "variant": asdict(probe.variant),
            }
            for probe in decision.probes
        ],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(sock.getsockname()[1])
    raise RuntimeError("No local TCP port is available for llama-server.")


def _base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
    raw_context = os.environ.get("MMM_LLAMA_SERVER_CTX", "").strip()
    context = 0
    if raw_context:
        try:
            value = int(raw_context)
        except ValueError:
            value = 0
        if value >= 0:
            context = value
    batch = _env_int("MMM_LLAMA_BATCH", 2048)
    ubatch = min(batch, _env_int("MMM_LLAMA_UBATCH", 512))
    kv = os.environ.get("MMM_KV_CACHE_QUANT", "q4_0").strip().lower() or "q4_0"
    return [
        binary,
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context),
        "--batch-size",
        str(batch),
        "--ubatch-size",
        str(ubatch),
        "--gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--cache-type-k",
        kv,
        "--cache-type-v",
        kv,
        "--load-mode",
        "none",
        # Tool-capable OpenAI chat requests require the Jinja chat engine.
        # This belongs to the server launch contract itself because autotune,
        # planner/coder priming and adapters can all be the first launch owner.
        "--jinja",
        # Keep Jinja prompt rendering (including the current tool schemas) but return
        # reasoning/tool markup in message.content. MMM validates and parses that raw
        # model protocol on the host instead of accepting llama.cpp-parsed tool calls.
        "--skip-chat-parsing",
        # Output-exhausted non-thinking actions are resumed by appending the exact
        # partial assistant turn. Pin the server capability explicitly rather than
        # depending on a build default that could change across Colab upgrades.
        "--prefill-assistant",
        "--no-ui",
        "--log-disable",
    ]


def _variant_args(variant: ServerVariant) -> list[str]:
    if variant.spec_type == "none":
        return ["--spec-type", "none"]
    return [
        "--spec-type",
        variant.spec_type,
        "--spec-draft-n-max",
        str(variant.draft_n_max),
        "--spec-draft-n-min",
        "0",
        "--spec-draft-ngl",
        "all",
    ]


def _start_server(
    binary: str,
    model_path: str,
    config: Any,
    variant: ServerVariant,
    port: int,
) -> subprocess.Popen[bytes]:
    debug = _env_bool("MMM_LLAMA_AUTOTUNE_DEBUG", False)
    stream = None if debug else subprocess.DEVNULL
    return subprocess.Popen(
        _base_args(binary, model_path, config, port) + _variant_args(variant),
        stdout=stream,
        stderr=stream,
    )


def _stop_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_ready(process: subprocess.Popen[bytes], port: int) -> str:
    import httpx

    timeout = _env_int("MMM_LLAMA_SERVER_START_TIMEOUT", 300)
    deadline = time.monotonic() + timeout
    origin = f"http://127.0.0.1:{port}"
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}.")
        try:
            response = httpx.get(f"{origin}/health", timeout=1.0)
            if response.status_code == 200:
                return f"{origin}/v1"
            last_error = f"health status={response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(last_error)


def _assistant_output(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    reasoning = str(message.get("reasoning_content") or "")
    content = str(message.get("content") or "")
    return reasoning + "\n<MMM-CONTENT>\n" + content


def _compact_benchmark_request(request: Any) -> Any:
    """Benchmark decode/MTP without repeatedly prefilling the real workflow prompt."""

    del request
    return SimpleNamespace(
        messages=(
            {
                "role": "system",
                "content": (
                    "Deterministic inference benchmark. Do not explain or reason. "
                    "Return only the requested JSON-like text."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Emit one object {"values":[0,1,2,...,63]} with every integer '
                    "from 0 through 63 in ascending order and no other text."
                ),
            },
        ),
        response_format="text",
    )


def _probe_server(
    base_url: str,
    request: Any,
    *,
    max_tokens: int,
    variant: ServerVariant,
) -> ProbeResult:
    import httpx

    payload: dict[str, Any] = {
        "model": "local",
        "messages": [dict(message) for message in request.messages],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 1234,
        "cache_prompt": False,
        "stream": False,
    }
    if getattr(request, "response_format", None) == "json":
        payload["response_format"] = {"type": "json_object"}

    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=_env_int("MMM_LLAMA_AUTOTUNE_REQUEST_TIMEOUT", 300),
        )
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - started
        output = _assistant_output(data)
        timings = data.get("timings") or {}
        usage = data.get("usage") or {}
        predicted_tokens = int(
            timings.get("predicted_n") or usage.get("completion_tokens") or 0
        )
        predicted_tps = float(timings.get("predicted_per_second") or 0.0)
        if predicted_tps <= 0 and predicted_tokens > 0 and elapsed > 0:
            predicted_tps = predicted_tokens / elapsed
        return ProbeResult(
            variant=variant,
            ok=bool(output) and predicted_tokens > 0,
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            predicted_tokens=predicted_tokens,
            predicted_tps=predicted_tps,
            prompt_tps=float(timings.get("prompt_per_second") or 0.0),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        return ProbeResult(
            variant=variant,
            ok=False,
            output_sha256="",
            predicted_tokens=0,
            predicted_tps=0.0,
            prompt_tps=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _benchmark(
    binary: str,
    model_path: str,
    config: Any,
    request: Any,
    fingerprint: str,
) -> AutotuneDecision | None:
    benchmark_request = _compact_benchmark_request(request)
    probe_tokens = _env_int(
        "MMM_LLAMA_AUTOTUNE_TOKENS", _BENCHMARK_OUTPUT_TOKENS
    )
    preferred_port = _env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910)
    probes: list[ProbeResult] = []

    # Width selection remains exhaustive. Only irrelevant prompt-prefill work is cut.
    for variant in _candidate_variants():
        port = _free_port(preferred_port)
        process: subprocess.Popen[bytes] | None = None
        try:
            process = _start_server(binary, model_path, config, variant, port)
            url = _wait_ready(process, port)
            _probe_server(url, benchmark_request, max_tokens=1, variant=variant)
            probes.append(
                _probe_server(
                    url,
                    benchmark_request,
                    max_tokens=probe_tokens,
                    variant=variant,
                )
            )
        except Exception as exc:
            probes.append(
                ProbeResult(
                    variant=variant,
                    ok=False,
                    output_sha256="",
                    predicted_tokens=0,
                    predicted_tps=0.0,
                    prompt_tps=0.0,
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            _stop_server(process)

    decision = _choose_variant(
        probes,
        minimum_speedup=_env_float("MMM_LLAMA_AUTOTUNE_MIN_SPEEDUP", 1.03),
    )
    if decision is None:
        return None
    return AutotuneDecision(
        fingerprint=fingerprint,
        selected=decision.selected,
        baseline_tps=decision.baseline_tps,
        selected_tps=decision.selected_tps,
        speedup=decision.speedup,
        probes=decision.probes,
    )


def _external_server_is_ready() -> bool:
    explicit = os.environ.get("LLAMA_SERVER_URL", "").strip()
    if not explicit:
        return False
    try:
        import httpx

        origin = explicit.removesuffix("/v1").rstrip("/")
        for endpoint in ("/v1/models", "/healthz", "/health"):
            try:
                if httpx.get(f"{origin}{endpoint}", timeout=0.5).status_code == 200:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _launch_selected(
    binary: str,
    model_path: str,
    config: Any,
    selected: ServerVariant,
) -> str:
    global _MANAGED_PROCESS, _MANAGED_URL

    preferred = _env_int("MMM_LLAMA_SERVER_PORT", 8910)
    port = _free_port(preferred)
    process = _start_server(binary, model_path, config, selected, port)
    try:
        url = _wait_ready(process, port)
    except Exception:
        _stop_server(process)
        raise
    _MANAGED_PROCESS = process
    _MANAGED_URL = url
    os.environ["LLAMA_SERVER_URL"] = url
    return url


def _baseline_decision(fingerprint: str) -> AutotuneDecision:
    return AutotuneDecision(
        fingerprint=fingerprint,
        selected=ServerVariant("baseline"),
        baseline_tps=0.0,
        selected_tps=0.0,
        speedup=1.0,
        probes=(),
    )


def _release_recoverable_attempt(fingerprint: str, exc: BaseException) -> None:
    """Allow a later retry only for explicitly classified transient resource failures."""
    if bool(getattr(exc, "_mmm_recoverable_resource_failure", False)):
        _ATTEMPTED_KEYS.discard(fingerprint)


def ensure_tuned_server(config: Any, request: Any) -> str:
    """Start one managed native server and never fall back to a second GGUF engine."""
    global _MANAGED_KEY

    if _external_server_is_ready():
        explicit = os.environ.get("LLAMA_SERVER_URL", "").strip()
        if explicit:
            return explicit

    binary = _server_binary()
    if binary is None:
        raise RuntimeError("native llama-server binary is unavailable")

    with _AUTOTUNE_LOCK:
        if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is None:
            if _MANAGED_URL:
                return _MANAGED_URL
            raise RuntimeError("managed llama-server process has no URL")

        model_path = _resolve_model_path(config)
        fingerprint = _fingerprint(config, binary, model_path)
        if fingerprint in _ATTEMPTED_KEYS:
            raise RuntimeError(
                "native llama-server startup was already attempted and did not leave "
                "a healthy managed process for this exact runtime fingerprint"
            )
        _ATTEMPTED_KEYS.add(fingerprint)

        try:
            decision = _load_cached_decision(fingerprint)
            if decision is None:
                if _env_bool("MMM_LLAMA_SERVER_AUTOTUNE", True):
                    decision = _benchmark(
                        binary, model_path, config, request, fingerprint
                    )
                    if decision is None:
                        raise RuntimeError(
                            "llama-server autotune could not validate a baseline decode"
                        )
                    _save_decision(decision)
                else:
                    decision = _baseline_decision(fingerprint)

            url = _launch_selected(binary, model_path, config, decision.selected)
            _MANAGED_KEY = fingerprint
            return url
        except Exception as exc:
            _release_recoverable_attempt(fingerprint, exc)
            raise


def _shutdown_managed_server() -> None:
    global _MANAGED_KEY, _MANAGED_PROCESS, _MANAGED_URL
    with _AUTOTUNE_LOCK:
        managed_key = _MANAGED_KEY
        _stop_server(_MANAGED_PROCESS)
        _MANAGED_PROCESS = None
        _MANAGED_URL = None
        _MANAGED_KEY = None
        if managed_key:
            _ATTEMPTED_KEYS.discard(managed_key)


atexit.register(_shutdown_managed_server)


def install() -> None:
    """Compatibility no-op; hardware policy exclusively owns adapter binding."""


__all__ = [
    "AutotuneDecision",
    "ProbeResult",
    "ServerVariant",
    "_base_args",
    "_benchmark",
    "_candidate_variants",
    "_choose_variant",
    "_compact_benchmark_request",
    "_probe_server",
    "_release_recoverable_attempt",
    "_server_binary",
    "_shutdown_managed_server",
    "_variant_args",
    "ensure_tuned_server",
    "install",
]
