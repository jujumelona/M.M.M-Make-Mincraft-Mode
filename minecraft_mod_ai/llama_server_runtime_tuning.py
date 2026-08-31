from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_SCHEMA_VERSION = "mmm/llama-server-autotune-v9-bounded-validated-parallel-context"
_INSTALL_LOCK = threading.RLock()
_MIB = 1024 * 1024
_MAX_PARALLEL = 8
_MAX_TOTAL_CONTEXT = 2_147_483_647
_STARTUP_LOG_LIMIT = 64 * 1024


@dataclass(frozen=True)
class ServerVariant:
    name: str
    spec_type: str = "none"
    draft_n_max: int = 0
    ubatch: int = 0
    parallel: int = 1
    cache_reuse: int = 0
    draft_p_min: float = 0.0


@dataclass(frozen=True)
class RuntimeResources:
    gpu_free_bytes: int = 0
    gpu_total_bytes: int = 0
    ram_available_bytes: int = 0
    cpu_count: int = 0


class RecoverableResourceLaunchError(RuntimeError):
    _mmm_recoverable_resource_failure = True


class _StartupLogBuffer:
    def __init__(self, limit: int = _STARTUP_LOG_LIMIT):
        self.limit = max(1024, int(limit))
        self.data = bytearray()
        self.lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self.lock:
            self.data.extend(chunk)
            if len(self.data) > self.limit:
                del self.data[: len(self.data) - self.limit]

    def text(self) -> str:
        with self.lock:
            value = bytes(self.data)
        return value.decode("utf-8", errors="replace")


def _attach_startup_log(process: Any) -> None:
    pipe = getattr(process, "stderr", None)
    if pipe is None or not callable(getattr(pipe, "read", None)):
        return
    buffer = _StartupLogBuffer()

    def drain() -> None:
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    return
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                buffer.append(chunk)
        except (OSError, ValueError):
            return

    reader = threading.Thread(
        target=drain, name="mmm_llama_startup_log", daemon=True
    )
    process._mmm_startup_log = buffer
    process._mmm_startup_log_reader = reader
    reader.start()


def _startup_log_tail(process: Any) -> str:
    reader = getattr(process, "_mmm_startup_log_reader", None)
    if reader is not None and getattr(process, "poll", lambda: None)() is not None:
        reader.join(timeout=0.25)
    buffer = getattr(process, "_mmm_startup_log", None)
    if not isinstance(buffer, _StartupLogBuffer):
        return ""
    raw = buffer.text()[-16_384:]
    lowered = raw.lower()
    if any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda_error_out_of_memory",
            "cannot allocate memory",
            "memory allocation failed",
        )
    ):
        marker = "out_of_memory"
    elif any(marker in lowered for marker in ("exit code 137", "signal 9", "sigkill")):
        marker = "resource_exit"
    else:
        marker = "none"
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    return f"resource_marker={marker} stderr_sha256={digest} captured_chars={len(raw)}"


def _performance_mode() -> str:
    raw = os.environ.get("MMM_PERFORMANCE_MODE", "").strip().lower()
    if not raw:
        raw = os.environ.get("MMM_LLAMA_TUNING_OBJECTIVE", "auto").strip().lower()
    if raw in {"latency", "single_stream", "single-stream"}:
        return "latency"
    if raw in {"throughput", "aggregate", "concurrent"}:
        return "throughput"
    return "auto"


def _optional_env_int(name: str, *, minimum: int = 0) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _cache_ram_mib() -> int:
    for name in ("MMM_LLAMA_CACHE_RAM_MIB", "MMM_LLAMA_CACHE_RAM"):
        value = _optional_env_int(name)
        if value is not None:
            return min(65_536, value)
    return 1024


def _receipt_number(payload: Any, names: set[str]) -> float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in names and isinstance(value, (int, float)):
                return float(value)
        for value in payload.values():
            found = _receipt_number(value, names)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = _receipt_number(value, names)
            if found is not None:
                return found
    return None


def _receipt_bytes(payload: Any, stem: str) -> int:
    bytes_value = _receipt_number(payload, {f"{stem}_bytes", f"{stem}_b"})
    if bytes_value is not None:
        return max(0, int(bytes_value))
    mib_value = _receipt_number(
        payload,
        {f"{stem}_mib", f"{stem}_mb", f"{stem}_memory_mib", f"{stem}_memory_mb"},
    )
    if mib_value is not None:
        return max(0, int(mib_value * _MIB))
    gib_value = _receipt_number(
        payload,
        {f"{stem}_gib", f"{stem}_gb", f"{stem}_memory_gib", f"{stem}_memory_gb"},
    )
    return max(0, int(gib_value * 1024 * _MIB)) if gib_value is not None else 0


def _setup_receipt() -> Any:
    raw = os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _runtime_resources() -> RuntimeResources:
    receipt = _setup_receipt()
    gpu_free = 0
    gpu_total = 0
    gpu_probe_succeeded = False
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        first = completed.stdout.strip().splitlines()[0]
        free_mib, total_mib = (int(token.strip()) for token in first.split(",")[:2])
        gpu_free = max(0, free_mib) * _MIB
        gpu_total = max(0, total_mib) * _MIB
        gpu_probe_succeeded = True
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        gpu_probe_succeeded = False

    ram_available = 0
    ram_probe_succeeded = False
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                ram_available = max(0, int(line.split()[1])) * 1024
                ram_probe_succeeded = True
                break
    except (OSError, UnicodeError, IndexError, ValueError):
        ram_probe_succeeded = False

    gpu_free_override = _optional_env_int("MMM_LLAMA_GPU_FREE_MIB")
    gpu_total_override = _optional_env_int("MMM_LLAMA_GPU_TOTAL_MIB")
    ram_override = _optional_env_int("MMM_LLAMA_RAM_AVAILABLE_MIB")
    if gpu_free_override is not None:
        gpu_free = gpu_free_override * _MIB
    elif not gpu_probe_succeeded:
        gpu_free = _receipt_bytes(receipt, "gpu_free") or _receipt_bytes(receipt, "vram_free")
    if gpu_total_override is not None:
        gpu_total = gpu_total_override * _MIB
    elif not gpu_probe_succeeded:
        gpu_total = _receipt_bytes(receipt, "gpu_total") or _receipt_bytes(receipt, "vram_total")
    if ram_override is not None:
        ram_available = ram_override * _MIB
    elif not ram_probe_succeeded:
        ram_available = (
            _receipt_bytes(receipt, "ram_available")
            or _receipt_bytes(receipt, "system_ram_available")
            or _receipt_bytes(receipt, "system_memory_available")
        )
    cpu_count = max(0, int(os.cpu_count() or 0))
    receipt_cpu = _receipt_number(receipt, {"cpu_count", "logical_cpu_count", "cpu_threads"})
    if not cpu_count and receipt_cpu is not None:
        cpu_count = max(0, int(receipt_cpu))
    return RuntimeResources(gpu_free, gpu_total, ram_available, cpu_count)


def _config_extra(config: Any) -> dict[str, Any]:
    extra = getattr(config, "extra", {})
    return dict(extra) if isinstance(extra, dict) else {}


def _is_qwen35_mtp_config(config: Any) -> bool:
    extra = _config_extra(config)
    return (
        str(extra.get("runtime_contract", "")).strip().casefold() == "qwen"
        and str(extra.get("decode_hotpath", "")).strip().casefold() == "t4_mtp"
    )


def _per_request_context(config: Any) -> int:
    names = (
        ("MMM_QWEN35_MTP_CTX", "MMM_LLAMA_SERVER_CTX")
        if _is_qwen35_mtp_config(config)
        else ("MMM_LLAMA_SERVER_CTX",)
    )
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        if value > 0:
            return value

    runtime_default = _config_extra(config).get("runtime_context_default")
    if runtime_default is not None:
        try:
            value = int(runtime_default)
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime_context_default must be a positive integer") from exc
        if value <= 0:
            raise ValueError("runtime_context_default must be a positive integer")
        try:
            capacity = int(getattr(config, "max_context", 0) or 0)
        except (TypeError, ValueError):
            capacity = 0
        return min(value, capacity) if capacity > 0 else value

    try:
        return max(0, int(getattr(config, "max_context", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _total_context(per_request: int, slots: int) -> int:
    slots = max(1, int(slots))
    if slots > _MAX_PARALLEL:
        raise RuntimeError(
            f"llama-server parallel slots {slots} exceeds supported maximum {_MAX_PARALLEL}"
        )
    if per_request <= 0 and slots > 1:
        raise RuntimeError(
            "parallel llama-server requires a positive per-request context so slots do not share an unknown context"
        )
    total = max(0, int(per_request)) * slots
    if total > _MAX_TOTAL_CONTEXT:
        raise RuntimeError(
            f"llama-server total context {total} exceeds supported maximum {_MAX_TOTAL_CONTEXT}"
        )
    return total


def _context_from_args(args: list[str], config: Any) -> int:
    for name in ("--ctx-size", "-c"):
        if name not in args:
            continue
        index = args.index(name)
        if index + 1 >= len(args):
            raise RuntimeError(f"{name} requires a context value")
        try:
            value = int(args[index + 1])
        except ValueError as exc:
            raise RuntimeError(f"invalid llama-server context: {args[index + 1]!r}") from exc
        if value < 0:
            raise RuntimeError("llama-server context must be non-negative")
        if value == 0:
            return _per_request_context(config)
        return value
    return _per_request_context(config)


def _kv_bytes_per_token() -> int:
    explicit = _optional_env_int("MMM_LLAMA_KV_BYTES_PER_TOKEN", minimum=1)
    if explicit is not None:
        return explicit
    cache_k = os.environ.get(
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_K", os.environ.get("MMM_KV_CACHE_QUANT", "q4_0")
    ).strip().lower()
    cache_v = os.environ.get(
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_V", os.environ.get("MMM_KV_CACHE_QUANT", "q4_0")
    ).strip().lower()
    per_half = {"q4_0": 24 * 1024, "q8_0": 40 * 1024, "f16": 80 * 1024}
    return per_half.get(cache_k, 40 * 1024) + per_half.get(cache_v, 40 * 1024)


def _model_size(model_path: str | None) -> int:
    if not model_path:
        return 0
    try:
        return max(0, int(Path(model_path).stat().st_size))
    except OSError:
        return 0


def _parallel_resource_feasible(
    slots: int,
    config: Any,
    model_path: str | None,
    resources: RuntimeResources,
) -> bool:
    slots = max(1, int(slots))
    if slots > _MAX_PARALLEL:
        return False
    if slots == 1:
        return True
    context = _per_request_context(config)
    try:
        total_context = _total_context(context, slots)
    except RuntimeError:
        return False
    model_bytes = _model_size(model_path) or 6 * _MIB * 1024
    gpu_free = resources.gpu_free_bytes or (14 * _MIB * 1024 if resources.gpu_total_bytes else 0)
    ram_avail = resources.ram_available_bytes or (12 * _MIB * 1024)
    if not gpu_free or not ram_avail:
        return False
    gpu_required = int(model_bytes * 1.05) + total_context * _kv_bytes_per_token() + 512 * _MIB
    ram_required = int(model_bytes * 0.30) + (512 + 256 * slots) * _MIB
    return bool(
        gpu_required <= int(gpu_free * 0.95)
        and ram_required <= int(ram_avail * 0.95)
    )


def _recoverable_resource_failure(
    failures: Iterable[str],
    *,
    slots: int,
    config: Any,
    model_path: str,
    resources: RuntimeResources,
) -> bool:
    rendered = " ".join(str(value).lower() for value in failures)
    if any(
        token in rendered
        for token in (
            "out of memory",
            "cannot allocate memory",
            "memory allocation",
            "cuda_error_out_of_memory",
            "resource_marker=out_of_memory",
        )
    ):
        return True
    resource_exit_evidence = any(
        token in rendered
        for token in (
            "exit code 137",
            "exited with code 137",
            "status 137",
            "signal 9",
            "sigkill",
            "resource_marker=resource_exit",
            "resource exhausted",
            "resource temporarily unavailable",
        )
    )
    if not resource_exit_evidence:
        return False
    model_bytes = _model_size(model_path)
    context = _per_request_context(config)
    if not model_bytes or not resources.gpu_free_bytes or not resources.ram_available_bytes:
        return False
    try:
        total_context = _total_context(context, max(1, slots))
    except RuntimeError:
        return False
    gpu_required = int(model_bytes * 1.07) + total_context * _kv_bytes_per_token() + 1280 * _MIB
    ram_required = int(model_bytes * 0.40) + (512 + 256 * max(1, slots)) * _MIB
    return bool(
        gpu_required > int(resources.gpu_free_bytes * 0.92)
        or ram_required > int(resources.ram_available_bytes * 0.90)
    )


def _resource_bucket(resources: RuntimeResources) -> dict[str, int]:
    def mib_bucket(value: int) -> int:
        mib = max(0, int(value)) // _MIB
        return (mib // 512) * 512

    return {
        "gpu_free_mib": mib_bucket(resources.gpu_free_bytes),
        "gpu_total_mib": mib_bucket(resources.gpu_total_bytes),
        "ram_available_mib": mib_bucket(resources.ram_available_bytes),
        "cpu_count": max(0, int(resources.cpu_count)),
    }


def _selection_inputs(config: Any) -> dict[str, Any]:
    tracked_env = (
        "MMM_LLAMA_PARALLEL",
        "MMM_LLAMA_CONCURRENT_REQUESTS",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
        "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
        "MMM_QWEN35_MTP_TUNING",
        "MMM_QWEN35_MTP_HOTPATH",
    )
    extra = _config_extra(config)
    return {
        "performance_mode": _performance_mode(),
        "parallel_target": _parallel_target(),
        "context_per_slot": _per_request_context(config),
        "model_id": str(getattr(config, "model_id", "")),
        "gguf_filename": str(extra.get("gguf_filename", "")),
        "max_new_tokens": int(getattr(config, "max_new_tokens", 0) or 0),
        "cache_ram_mib": _cache_ram_mib(),
        "env": {name: os.environ.get(name, "") for name in tracked_env},
    }


def _json_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _medium_prefill_request(request: Any) -> Any:
    messages = tuple(getattr(request, "messages", ()) or ())
    medium_context = (
        "Synthetic benchmark context (ignore its content when answering): "
        + " registry dependency event network asset recipe validation" * 256
    )
    replacement_messages = messages + ({"role": "user", "content": medium_context},)
    try:
        return replace(request, messages=replacement_messages)
    except (TypeError, ValueError):
        values = dict(vars(request)) if hasattr(request, "__dict__") else {}
        values["messages"] = replacement_messages
        values.setdefault("response_format", getattr(request, "response_format", "text"))
        return SimpleNamespace(**values)


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


def _parse_int_candidates(raw: str, *, minimum: int, maximum: int, required: Iterable[int] = ()) -> tuple[int, ...]:
    values: list[int] = []
    for value in required:
        value = int(value)
        if minimum <= value <= maximum and value not in values:
            values.append(value)
    for token in raw.split(","):
        try:
            value = int(token.strip())
        except ValueError:
            continue
        if minimum <= value <= maximum and value not in values:
            values.append(value)
    return tuple(values)


def _replace_option(args: list[str], names: tuple[str, ...], value: str) -> None:
    for name in names:
        try:
            index = args.index(name)
        except ValueError:
            continue
        if index + 1 < len(args):
            args[index + 1] = value
            return
    args.extend([names[0], value])


def _remove_option(args: list[str], names: tuple[str, ...], *, takes_value: bool = True) -> None:
    for name in names:
        while name in args:
            index = args.index(name)
            del args[index]
            if takes_value and index < len(args):
                del args[index]


def _ubatch_candidates(autotune_module: Any) -> tuple[int, ...]:
    batch = autotune_module._env_int("MMM_LLAMA_BATCH", 2048)
    current = min(batch, autotune_module._env_int("MMM_LLAMA_UBATCH", 512))
    return _parse_int_candidates(
        os.environ.get("MMM_LLAMA_UBATCH_CANDIDATES", "512,1024,2048"),
        minimum=64,
        maximum=max(64, batch),
        required=(current,),
    )


def _cache_reuse_candidates() -> tuple[int, ...]:
    return _parse_int_candidates(
        os.environ.get("MMM_LLAMA_CACHE_REUSE_CANDIDATES", "0,64,256"),
        minimum=0,
        maximum=8192,
        required=(0,),
    )


def _parallel_target() -> int:
    explicit = os.environ.get("MMM_LLAMA_CONCURRENT_REQUESTS", "").strip()
    if explicit:
        return _env_int(
            "MMM_LLAMA_CONCURRENT_REQUESTS", 1, minimum=1, maximum=_MAX_PARALLEL
        )
    return 1 if _performance_mode() == "latency" else _MAX_PARALLEL


def _explicit_parallel() -> int | None:
    raw = os.environ.get("MMM_LLAMA_PARALLEL", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("MMM_LLAMA_PARALLEL must be an integer") from exc
    return max(1, min(_MAX_PARALLEL, value))


def _parallel_candidates(
    config: Any | None = None,
    model_path: str | None = None,
    resources: RuntimeResources | None = None,
) -> tuple[int, ...]:
    explicit = _explicit_parallel()
    if explicit is not None:
        return (explicit,)
    explicit_concurrency = os.environ.get("MMM_LLAMA_CONCURRENT_REQUESTS", "").strip()
    target = _parallel_target()
    if _performance_mode() == "latency" and not explicit_concurrency:
        return (1,)
    values = tuple(value for value in (1, 2, 4, 8) if value <= target)
    if explicit_concurrency:
        return values or (1,)
    if config is None:
        return (1,)
    snapshot = resources or _runtime_resources()
    feasible = tuple(
        value
        for value in values
        if _parallel_resource_feasible(value, config, model_path, snapshot)
    )
    return feasible or (1,)


def _candidate_variants(autotune_module: Any) -> tuple[ServerVariant, ...]:
    del autotune_module
    base_widths = _parse_int_candidates(
        os.environ.get("MMM_LLAMA_MTP_WIDTHS", "1,2,3"), minimum=1, maximum=32
    )
    confidence_widths = _parse_int_candidates(
        os.environ.get("MMM_LLAMA_MTP_CONFIDENCE_WIDTHS", "2,4,8,16"), minimum=1, maximum=32
    )
    try:
        confidence_p_min = float(os.environ.get("MMM_LLAMA_MTP_SEED_P_MIN", "0.8"))
    except ValueError:
        confidence_p_min = 0.8
    confidence_p_min = max(0.0, min(0.999, confidence_p_min))

    values = [ServerVariant("baseline")]
    values.extend(ServerVariant(f"mtp-{w}", "draft-mtp", w) for w in base_widths)
    if confidence_p_min > 0:
        for width in confidence_widths:
            candidate = ServerVariant(
                f"mtp-{width}|pm{confidence_p_min:g}",
                "draft-mtp",
                width,
                draft_p_min=confidence_p_min,
            )
            if all(
                not (
                    item.spec_type == "draft-mtp"
                    and item.draft_n_max == candidate.draft_n_max
                    and float(getattr(item, "draft_p_min", 0.0)) == candidate.draft_p_min
                )
                for item in values
            ):
                values.append(candidate)

    allowed = {"ngram-simple", "ngram-mod", "ngram-map-k", "ngram-map-k4v", "ngram-cache"}
    for token in os.environ.get("MMM_LLAMA_NGRAM_SPEC_TYPES", "ngram-simple,ngram-mod,ngram-map-k").split(","):
        spec_type = token.strip()
        if spec_type in allowed and all(item.spec_type != spec_type for item in values):
            values.append(ServerVariant(spec_type, spec_type))
    return tuple(values)


def _model_supports_mtp(config: Any) -> bool:
    extra = getattr(config, "extra", {})
    if isinstance(extra, dict) and "supports_mtp" in extra:
        return bool(extra.get("supports_mtp"))
    model_id = str(getattr(config, "model_id", "")).upper()
    filename = str(extra.get("gguf_filename", "")).upper() if isinstance(extra, dict) else ""
    return "-MTP-GGUF" in model_id or "-MTP-" in filename


def _candidate_variants_for_config(autotune_module: Any, config: Any) -> tuple[ServerVariant, ...]:
    values = _candidate_variants(autotune_module)
    if _model_supports_mtp(config):
        return values
    return tuple(value for value in values if value.spec_type != "draft-mtp")


def _eligible(probe: Any, baseline: Any) -> bool:
    return (
        bool(getattr(probe, "ok", False))
        and str(getattr(probe, "output_sha256", ""))
        == str(getattr(baseline, "output_sha256", ""))
        and float(getattr(probe, "predicted_tps", 0.0)) > 0
    )


def _balanced_score(probe: Any, baseline: Any) -> float:
    decode_base = max(1e-9, float(getattr(baseline, "predicted_tps", 0.0)))
    decode = max(0.0, float(getattr(probe, "predicted_tps", 0.0))) / decode_base
    prompt_base = float(getattr(baseline, "prompt_tps", 0.0))
    prompt_value = float(getattr(probe, "prompt_tps", 0.0))
    prompt = prompt_value / prompt_base if prompt_base > 0 and prompt_value > 0 else 1.0
    return 0.70 * decode + 0.30 * prompt


def _select_probe(probes: list[Any], *, balanced: bool, minimum_gain: float) -> Any | None:
    if not probes:
        return None
    baseline = probes[0]
    if not getattr(baseline, "ok", False) or float(getattr(baseline, "predicted_tps", 0.0)) <= 0:
        return None
    valid = [baseline] + [probe for probe in probes[1:] if _eligible(probe, baseline)]
    score = (
        (lambda probe: _balanced_score(probe, baseline))
        if balanced
        else (
            lambda probe: float(getattr(probe, "predicted_tps", 0.0))
            / max(1e-9, float(getattr(baseline, "predicted_tps", 0.0)))
        )
    )
    best = max(valid, key=score)
    return best if best is baseline or score(best) >= max(1.0, minimum_gain) else baseline


def _select_parallel_probe(
    reference: Any, probes: Iterable[Any], *, minimum_gain: float
) -> Any:
    eligible = [probe for probe in probes if _eligible(probe, reference)]
    if not eligible:
        return reference
    best = max(eligible, key=lambda probe: float(getattr(probe, "predicted_tps", 0.0)))
    required = float(getattr(reference, "predicted_tps", 0.0)) * max(1.0, minimum_gain)
    return best if float(getattr(best, "predicted_tps", 0.0)) >= required else reference


def _parallel_probe(
    autotune_module: Any,
    base_url: str,
    request: Any,
    *,
    max_tokens: int,
    variant: ServerVariant,
    concurrency: int,
) -> Any:
    started = time.perf_counter()
    try:
        rounds: list[list[Any]] = []
        for probe_request in (request, _medium_prefill_request(request)):
            with ThreadPoolExecutor(
                max_workers=concurrency, thread_name_prefix="mmm_llama_slot_probe"
            ) as pool:
                values = [
                    future.result()
                    for future in [
                        pool.submit(
                            autotune_module._probe_server,
                            base_url,
                            probe_request,
                            max_tokens=max_tokens,
                            variant=variant,
                        )
                        for _ in range(concurrency)
                    ]
                ]
            rounds.append(values)
        elapsed = time.perf_counter() - started
        round_hashes: list[str] = []
        for values in rounds:
            round_first = values[0]
            if not all(getattr(value, "ok", False) for value in values) or any(
                getattr(value, "output_sha256", "")
                != getattr(round_first, "output_sha256", "")
                for value in values[1:]
            ):
                raise RuntimeError("parallel slot probe produced non-identical or failed outputs")
            round_hashes.append(str(getattr(round_first, "output_sha256", "")))
        tokens = sum(
            int(getattr(value, "predicted_tokens", 0))
            for values in rounds
            for value in values
        )
        prompt_values = [
            float(getattr(value, "prompt_tps", 0.0))
            for values in rounds
            for value in values
            if float(getattr(value, "prompt_tps", 0.0)) > 0
        ]
        combined_output_sha256 = hashlib.sha256(
            json.dumps(round_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return autotune_module.ProbeResult(
            variant=variant,
            ok=tokens > 0 and elapsed > 0,
            output_sha256=combined_output_sha256,
            predicted_tokens=tokens,
            predicted_tps=(tokens / elapsed if elapsed > 0 else 0.0),
            prompt_tps=(sum(prompt_values) / len(prompt_values) if prompt_values else 0.0),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark failures become comparable ProbeResults
        return autotune_module.ProbeResult(
            variant=variant,
            ok=False,
            output_sha256="",
            predicted_tokens=0,
            predicted_tps=0.0,
            prompt_tps=0.0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_parallel_stage(
    run_variant: Any,
    *,
    binary: str,
    model_path: str,
    config: Any,
    benchmark_request: Any,
    selected: ServerVariant,
    probe_tokens: int,
    parallel_values: Iterable[int],
    minimum_gain: float,
    forced_parallel: int | None = None,
) -> tuple[ServerVariant, Any | None, Any | None, tuple[Any, ...]]:
    values = tuple(sorted({int(value) for value in parallel_values}))
    if not any(value > 1 for value in values):
        return selected, None, None, ()
    root_name = selected.name.split("|p", 1)[0]
    p1_variant = replace(selected, name=f"{root_name}|p1", parallel=1)
    p1_probe = run_variant(
        binary,
        model_path,
        config,
        benchmark_request,
        p1_variant,
        probe_tokens=probe_tokens,
        parallel_probe=True,
        concurrency=1,
        propagate_resource_failure=True,
    )
    measured: list[Any] = [p1_probe]
    if not bool(getattr(p1_probe, "ok", False)) or float(
        getattr(p1_probe, "predicted_tps", 0.0)
    ) <= 0:
        if forced_parallel is not None and forced_parallel > 1:
            raise RuntimeError(
                f"explicit MMM_LLAMA_PARALLEL={forced_parallel} could not validate the p1 exact-output reference"
            )
        return selected, None, p1_probe, tuple(measured)
    candidates: list[Any] = []
    for value in values:
        if value <= 1:
            continue
        variant = replace(selected, name=f"{root_name}|p{value}", parallel=value)
        probe = run_variant(
            binary,
            model_path,
            config,
            benchmark_request,
            variant,
            probe_tokens=probe_tokens,
            parallel_probe=True,
            concurrency=value,
            propagate_resource_failure=(forced_parallel == value),
        )
        measured.append(probe)
        candidates.append(probe)
    winner = _select_parallel_probe(p1_probe, candidates, minimum_gain=minimum_gain)
    if forced_parallel is not None and forced_parallel > 1:
        forced = next(
            (
                probe
                for probe in candidates
                if int(getattr(getattr(probe, "variant", None), "parallel", 1))
                == forced_parallel
                and _eligible(probe, p1_probe)
            ),
            None,
        )
        if forced is None:
            raise RuntimeError(
                f"explicit MMM_LLAMA_PARALLEL={forced_parallel} failed exact-output runtime validation"
            )
        winner = forced
    return winner.variant, winner, p1_probe, tuple(measured)


def install(autotune_module: Any) -> None:
    with _INSTALL_LOCK:
        if getattr(autotune_module, "_mmm_runtime_tuning_installed", False):
            return
        autotune_module.ServerVariant = ServerVariant
        autotune_module._BENCHMARK_SCHEMA_VERSION = _SCHEMA_VERSION

        def candidate_variants() -> tuple[ServerVariant, ...]:
            return _candidate_variants(autotune_module)

        candidate_variants._mmm_mtp_ngram_autotune = True
        autotune_module._candidate_variants = candidate_variants

        current_base = autotune_module._base_args

        @wraps(current_base)
        def tuned_base_args(binary: str, model_path: str, config: Any, port: int) -> list[str]:
            args = list(current_base(binary, model_path, config, port))
            _replace_option(args, ("--load-mode", "-lm"), "auto")
            if "--cache-prompt" not in args:
                args.append("--cache-prompt")
            _replace_option(args, ("--cache-ram",), str(_cache_ram_mib()))
            return args

        for tag in (
            "_mmm_auto_gpu_layers",
            "_mmm_single_decode_slot",
            "_mmm_native_telemetry_endpoints",
        ):
            if getattr(current_base, tag, False):
                setattr(tuned_base_args, tag, True)
        tuned_base_args._mmm_load_mode_auto = True
        autotune_module._base_args = tuned_base_args

        current_variant_args = autotune_module._variant_args

        @wraps(current_variant_args)
        def tuned_variant_args(variant: ServerVariant) -> list[str]:
            if variant.spec_type.startswith("ngram-"):
                return ["--spec-type", variant.spec_type]
            return current_variant_args(variant)

        if getattr(current_variant_args, "_mmm_auto_draft_layers", False):
            tuned_variant_args._mmm_auto_draft_layers = True
        tuned_variant_args._mmm_ngram_speculation = True
        autotune_module._variant_args = tuned_variant_args

        def start_server(
            binary: str,
            model_path: str,
            config: Any,
            variant: ServerVariant,
            port: int,
        ) -> subprocess.Popen[bytes]:
            debug = autotune_module._env_bool("MMM_LLAMA_AUTOTUNE_DEBUG", False)
            args = list(autotune_module._base_args(binary, model_path, config, port))
            native_context = any(
                name in args
                and args.index(name) + 1 < len(args)
                and args[args.index(name) + 1] == "0"
                for name in ("--ctx-size", "-c")
            )
            slots = max(1, int(variant.parallel))
            per_request_context = _context_from_args(args, config)
            total_context = _total_context(per_request_context, slots)
            if slots > 1 or not native_context:
                _replace_option(args, ("--ctx-size", "-c"), str(total_context))
            if variant.ubatch > 0:
                _replace_option(args, ("--ubatch-size", "-ub"), str(variant.ubatch))
            _replace_option(args, ("--parallel", "-np"), str(slots))
            if slots > 1:
                if "--cont-batching" not in args and "-cb" not in args:
                    args.append("--cont-batching")
                if "--kv-unified" not in args and "-kvu" not in args:
                    args.append("--kv-unified")
            if "--cache-prompt" not in args:
                _remove_option(args, ("--cache-ram",), takes_value=True)
            _remove_option(args, ("--cache-reuse",), takes_value=True)
            if variant.cache_reuse > 0:
                args.extend(["--cache-reuse", str(variant.cache_reuse)])
            args.extend(autotune_module._variant_args(variant))
            process = subprocess.Popen(
                args,
                stdout=None if debug else subprocess.DEVNULL,
                stderr=None if debug else subprocess.PIPE,
            )
            if not debug:
                _attach_startup_log(process)
            return process

        start_server._mmm_staged_runtime_tuning = True
        autotune_module._start_server = start_server

        current_wait_ready = getattr(autotune_module, "_wait_ready", None)
        if callable(current_wait_ready):

            @wraps(current_wait_ready)
            def wait_ready(process: Any, port: int) -> str:
                try:
                    return current_wait_ready(process, port)
                except Exception as exc:
                    tail = _startup_log_tail(process)
                    if tail:
                        raise RuntimeError(
                            f"{exc}; bounded llama-server startup diagnostic: {tail}"
                        ) from exc
                    raise

            wait_ready._mmm_bounded_startup_diagnostics = True
            autotune_module._wait_ready = wait_ready

        current_fingerprint = autotune_module._fingerprint

        @wraps(current_fingerprint)
        def tuning_fingerprint(config: Any, binary: str, model_path: str) -> str:
            resources = _runtime_resources()
            parallel_candidates = _parallel_candidates(config, model_path, resources)
            prompt_cache_enabled = not (
                _is_qwen35_mtp_config(config)
                and os.environ.get("MMM_QWEN35_MTP_HOTPATH", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            payload = {
                "schema": _SCHEMA_VERSION,
                "base": current_fingerprint(config, binary, model_path),
                "mtp_supported": _model_supports_mtp(config),
                "load_mode": "auto",
                "spec_variants": [
                    asdict(value)
                    for value in _candidate_variants_for_config(autotune_module, config)
                ],
                "ubatch_candidates": _ubatch_candidates(autotune_module),
                "cache_reuse_candidates": _cache_reuse_candidates(),
                "parallel_candidates": parallel_candidates,
                "concurrent_requests": _parallel_target(),
                "performance_mode": _performance_mode(),
                "resource_bucket": _resource_bucket(resources),
                "context_per_slot": _per_request_context(config),
                "total_context_candidates": {
                    str(value): _total_context(_per_request_context(config), value)
                    for value in parallel_candidates
                },
                "prompt_cache": prompt_cache_enabled,
                "cache_ram_mib": _cache_ram_mib() if prompt_cache_enabled else 0,
                "search": "resource-feasible-p1-p2-p4-p8+adaptive-joint-mtp+ubatch",
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        if getattr(current_fingerprint, "_mmm_stable_model_signature", False):
            tuning_fingerprint._mmm_stable_model_signature = True
        tuning_fingerprint._mmm_runtime_tuning_fingerprint = True
        autotune_module._fingerprint = tuning_fingerprint

        def run_variant(
            binary: str,
            model_path: str,
            config: Any,
            benchmark_request: Any,
            variant: ServerVariant,
            *,
            probe_tokens: int,
            parallel_probe: bool = False,
            concurrency: int = 1,
            propagate_resource_failure: bool = False,
        ) -> Any:
            port = autotune_module._free_port(
                autotune_module._env_int("MMM_LLAMA_AUTOTUNE_PORT", 18910)
            )
            process = None
            try:
                process = autotune_module._start_server(
                    binary, model_path, config, variant, port
                )
                url = autotune_module._wait_ready(process, port)
                autotune_module._probe_server(
                    url, benchmark_request, max_tokens=1, variant=variant
                )
                if parallel_probe:
                    return _parallel_probe(
                        autotune_module,
                        url,
                        benchmark_request,
                        max_tokens=probe_tokens,
                        variant=variant,
                        concurrency=concurrency,
                    )
                return autotune_module._probe_server(
                    url,
                    benchmark_request,
                    max_tokens=probe_tokens,
                    variant=variant,
                )
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                if propagate_resource_failure and _recoverable_resource_failure(
                    (failure,),
                    slots=max(1, int(getattr(variant, "parallel", 1) or 1)),
                    config=config,
                    model_path=model_path,
                    resources=_runtime_resources(),
                ):
                    raise RecoverableResourceLaunchError(
                        "native llama-server tuning probe hit a transient resource failure "
                        "(resource_marker=out_of_memory_or_resource_exit)"
                    ) from exc
                return autotune_module.ProbeResult(
                    variant=variant,
                    ok=False,
                    output_sha256="",
                    predicted_tokens=0,
                    predicted_tps=0.0,
                    prompt_tps=0.0,
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                autotune_module._stop_server(process)

        autotune_module._mmm_run_tuning_variant = run_variant
        run_variant._mmm_resource_failure_propagation = True

        def benchmark(
            binary: str,
            model_path: str,
            config: Any,
            request: Any,
            fingerprint: str,
        ) -> Any | None:
            benchmark_request = autotune_module._compact_benchmark_request(request)
            probe_tokens = min(
                int(config.max_new_tokens),
                autotune_module._env_int(
                    "MMM_LLAMA_AUTOTUNE_TOKENS",
                    autotune_module._BENCHMARK_OUTPUT_TOKENS,
                ),
            )
            stage_gain = autotune_module._env_float("MMM_LLAMA_STAGE_MIN_GAIN", 1.01)
            spec_gain = max(
                stage_gain,
                autotune_module._env_float("MMM_LLAMA_AUTOTUNE_MIN_SPEEDUP", 1.01),
            )
            probes: list[Any] = []
            values = _candidate_variants_for_config(autotune_module, config)
            baseline_variant = next(value for value in values if value.spec_type == "none")
            baseline = run_variant(
                binary,
                model_path,
                config,
                benchmark_request,
                baseline_variant,
                probe_tokens=probe_tokens,
                propagate_resource_failure=True,
            )
            probes.append(baseline)
            if not getattr(baseline, "ok", False) or float(
                getattr(baseline, "predicted_tps", 0.0)
            ) <= 0:
                return None

            mtp_values = [value for value in values if value.spec_type == "draft-mtp"]
            ngram_values = [
                value for value in values if value.spec_type.startswith("ngram-")
            ]
            primary = mtp_values if _model_supports_mtp(config) else ngram_values
            primary_probes = [baseline]
            for variant in primary:
                probe = run_variant(
                    binary,
                    model_path,
                    config,
                    benchmark_request,
                    variant,
                    probe_tokens=probe_tokens,
                )
                probes.append(probe)
                primary_probes.append(probe)
            spec = (
                _select_probe(
                    primary_probes, balanced=False, minimum_gain=spec_gain
                )
                or baseline
            )

            if _model_supports_mtp(config) and spec is baseline and ngram_values:
                fallback_probes = [baseline]
                for variant in ngram_values:
                    probe = run_variant(
                        binary,
                        model_path,
                        config,
                        benchmark_request,
                        variant,
                        probe_tokens=probe_tokens,
                    )
                    probes.append(probe)
                    fallback_probes.append(probe)
                spec = (
                    _select_probe(
                        fallback_probes, balanced=False, minimum_gain=spec_gain
                    )
                    or baseline
                )

            selected = replace(
                spec.variant,
                ubatch=min(
                    autotune_module._env_int("MMM_LLAMA_BATCH", 2048),
                    autotune_module._env_int("MMM_LLAMA_UBATCH", 512),
                ),
            )
            final_probe = spec
            base_ubatch = selected.ubatch
            for value in sorted(
                v for v in _ubatch_candidates(autotune_module) if v != base_ubatch
            ):
                variant = replace(
                    selected,
                    name=f"{selected.name.split('|ub', 1)[0]}|ub{value}",
                    ubatch=value,
                )
                probe = run_variant(
                    binary,
                    model_path,
                    config,
                    benchmark_request,
                    variant,
                    probe_tokens=probe_tokens,
                )
                probes.append(probe)
                if not _eligible(probe, baseline):
                    continue
                if _balanced_score(probe, final_probe) >= max(1.0, stage_gain):
                    selected = variant
                    final_probe = probe

            resources = _runtime_resources()
            parallel_values = _parallel_candidates(config, model_path, resources)
            decision_baseline = baseline
            selected, parallel_winner, p1_probe, parallel_probes = _run_parallel_stage(
                run_variant,
                binary=binary,
                model_path=model_path,
                config=config,
                benchmark_request=benchmark_request,
                selected=selected,
                probe_tokens=probe_tokens,
                parallel_values=parallel_values,
                minimum_gain=stage_gain,
                forced_parallel=_explicit_parallel(),
            )
            probes.extend(parallel_probes)
            if parallel_winner is not None and p1_probe is not None:
                final_probe = parallel_winner
                decision_baseline = p1_probe

            baseline_tps = float(getattr(decision_baseline, "predicted_tps", 0.0))
            selected_tps = float(getattr(final_probe, "predicted_tps", 0.0))
            return autotune_module.AutotuneDecision(
                fingerprint=fingerprint,
                selected=selected,
                baseline_tps=baseline_tps,
                selected_tps=selected_tps,
                speedup=(selected_tps / baseline_tps if baseline_tps > 0 else 1.0),
                probes=tuple(probes),
            )

        benchmark._mmm_staged_runtime_tuning = True
        benchmark._mmm_model_eligible_speculation = True
        benchmark._mmm_adaptive_cold_search = True
        benchmark._mmm_adaptive_joint_mtp_search = True
        benchmark._mmm_exhaustive_ubatch_search = True
        autotune_module._benchmark = benchmark

        current_launch = autotune_module._launch_selected

        @wraps(current_launch)
        def launch_selected(
            binary: str, model_path: str, config: Any, selected: ServerVariant
        ) -> str:
            explicit_parallel = _explicit_parallel()
            requested_slots = (
                explicit_parallel
                if explicit_parallel is not None
                else max(1, int(getattr(selected, "parallel", 1) or 1))
            )
            if requested_slots > _MAX_PARALLEL:
                raise RuntimeError(
                    f"llama-server parallel slots {requested_slots} exceeds supported maximum {_MAX_PARALLEL}"
                )
            exact_parallel = explicit_parallel is not None
            if exact_parallel:
                attempts = [requested_slots]
            else:
                attempts = [requested_slots]
                attempts.extend(
                    value
                    for value in (8, 4, 2, 1)
                    if value < requested_slots and value not in attempts
                )

            failures: list[str] = []
            active = selected
            url = ""
            for slots in attempts:
                root_name = str(getattr(selected, "name", "baseline")).split(
                    "|p", 1
                )[0]
                active = replace(
                    selected,
                    name=root_name if slots == 1 else f"{root_name}|p{slots}",
                    parallel=slots,
                )
                try:
                    url = current_launch(binary, model_path, config, active)
                    break
                except Exception as exc:  # noqa: BLE001 - isolate sequential launch candidates
                    failures.append(f"p{slots}: {type(exc).__name__}: {exc}")
            if not url:
                message = (
                    "native llama-server failed every sequential slot launch: "
                    + " | ".join(failures)
                )
                fresh_resources = _runtime_resources()
                if _recoverable_resource_failure(
                    failures,
                    slots=min(attempts),
                    config=config,
                    model_path=model_path,
                    resources=fresh_resources,
                ):
                    raise RecoverableResourceLaunchError(message)
                raise RuntimeError(message)

            slots = max(1, active.parallel)
            context_per_slot = _per_request_context(config)
            context_total = _total_context(context_per_slot, slots)
            active_ubatch = active.ubatch or min(
                autotune_module._env_int("MMM_LLAMA_BATCH", 2048),
                autotune_module._env_int("MMM_LLAMA_UBATCH", 512),
            )
            kv_k = os.environ.get(
                "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
                os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"),
            ).strip().lower()
            kv_v = os.environ.get(
                "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
                os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"),
            ).strip().lower()
            prompt_cache_enabled = not (
                _is_qwen35_mtp_config(config)
                and os.environ.get("MMM_QWEN35_MTP_HOTPATH", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            resources = _runtime_resources()
            receipt = {
                "schema_version": "mmm/llama-runtime-receipt-v1",
                "performance_mode": _performance_mode(),
                "slots": slots,
                "context_per_slot": context_per_slot,
                "context_total": context_total,
                "ubatch": active_ubatch,
                "kv_k": kv_k,
                "kv_v": kv_v,
                "spec_type": active.spec_type,
                "draft_n_max": int(active.draft_n_max),
                "draft_p_min": float(active.draft_p_min),
                "cache_reuse": int(active.cache_reuse),
                "prompt_cache": prompt_cache_enabled,
                "cache_ram_mib": _cache_ram_mib() if prompt_cache_enabled else 0,
                "resource_bucket": _resource_bucket(resources),
            }
            receipt["selection_inputs_sha256"] = _json_fingerprint(
                _selection_inputs(config)
            )
            receipt["selection_sha256"] = _json_fingerprint(receipt)
            encoded_receipt = json.dumps(
                receipt, sort_keys=True, separators=(",", ":")
            )
            os.environ["MMM_LLAMA_RUNTIME_RECEIPT"] = encoded_receipt
            autotune_module._MMM_LLAMA_RUNTIME_RECEIPT = receipt
            os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = str(slots)
            os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(active_ubatch)
            os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = str(active.cache_reuse)
            os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = active.spec_type
            print(
                "native llama-server: runtime selected",
                f"slots={slots}",
                f"context={context_per_slot}x{slots}={context_total}",
                f"ubatch={active_ubatch}",
                f"kv={kv_k}/{kv_v}",
                f"spec={active.spec_type}:{int(active.draft_n_max)}",
                flush=True,
            )
            return url

        launch_selected._mmm_exports_active_runtime = True
        autotune_module._launch_selected = launch_selected

        current_ensure = autotune_module.ensure_tuned_server

        @wraps(current_ensure)
        def ensure_current_runtime(config: Any, request: Any) -> str:
            lock = getattr(autotune_module, "_AUTOTUNE_LOCK", None)
            if lock is None:
                return current_ensure(config, request)
            expected = _json_fingerprint(_selection_inputs(config))
            with lock:
                process = getattr(autotune_module, "_MANAGED_PROCESS", None)
                managed_key = getattr(autotune_module, "_MANAGED_KEY", None)
                managed_url = str(
                    getattr(autotune_module, "_MANAGED_URL", "") or ""
                ).rstrip("/")
                if process is None or process.poll() is not None or not managed_url:
                    return current_ensure(config, request)
                configured_url = os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
                if configured_url != managed_url:
                    return current_ensure(config, request)

                receipt = getattr(
                    autotune_module, "_MMM_LLAMA_RUNTIME_RECEIPT", None
                )
                if not isinstance(receipt, dict):
                    try:
                        receipt = json.loads(
                            os.environ.get("MMM_LLAMA_RUNTIME_RECEIPT", "")
                        )
                    except (json.JSONDecodeError, TypeError):
                        receipt = None
                receipt_sha = (
                    str(receipt.get("selection_inputs_sha256", ""))
                    if isinstance(receipt, dict)
                    else ""
                )
                owner_snapshot = (process, managed_key, managed_url, receipt_sha)
                if receipt_sha == expected:
                    return current_ensure(config, request)

                if owner_snapshot != (
                    getattr(autotune_module, "_MANAGED_PROCESS", None),
                    getattr(autotune_module, "_MANAGED_KEY", None),
                    str(getattr(autotune_module, "_MANAGED_URL", "") or "").rstrip("/"),
                    receipt_sha,
                ):
                    return current_ensure(config, request)
                autotune_module._shutdown_managed_server()
                if (
                    os.environ.get("LLAMA_SERVER_URL", "").strip().rstrip("/")
                    == managed_url
                ):
                    os.environ.pop("LLAMA_SERVER_URL", None)
                os.environ.pop("MMM_LLAMA_RUNTIME_RECEIPT", None)
                autotune_module._MMM_LLAMA_RUNTIME_RECEIPT = None
                return autotune_module.ensure_tuned_server(config, request)

        ensure_current_runtime._mmm_refreshes_stale_runtime_selection = True
        autotune_module.ensure_tuned_server = ensure_current_runtime
        autotune_module._mmm_runtime_tuning_installed = True


__all__ = [
    "RuntimeResources",
    "ServerVariant",
    "_cache_ram_mib",
    "_cache_reuse_candidates",
    "_candidate_variants_for_config",
    "_explicit_parallel",
    "_medium_prefill_request",
    "_model_supports_mtp",
    "_parallel_candidates",
    "_parallel_resource_feasible",
    "_parallel_target",
    "_parse_int_candidates",
    "_per_request_context",
    "_performance_mode",
    "_replace_option",
    "_resource_bucket",
    "_runtime_resources",
    "_total_context",
    "_ubatch_candidates",
    "install",
]
