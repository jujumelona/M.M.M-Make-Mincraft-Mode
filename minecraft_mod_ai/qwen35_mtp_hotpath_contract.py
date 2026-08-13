from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_MAX_CTX = 2147483647
_MARKER = "_mmm_qwen35_measured_decode_hotpath_v11"
_BASE_MARKER = "_mmm_qwen35_measured_fast_args_v7"
_VARIANT_MARKER = "_mmm_qwen35_full_draft_gpu_v2"
_FINGERPRINT_MARKER = "_mmm_qwen35_draft_gpu_fingerprint_v2"
_DRAFT_KV_BENCHMARK_MARKER = "_mmm_qwen35_draft_kv_benchmark_v1"
_DRAFT_KV_LAUNCH_MARKER = "_mmm_qwen35_draft_kv_launch_v1"
_SLOT_POLL_MARKER = "_mmm_qwen35_no_decode_slot_poll_v1"
_ACTIVE_TUNING_ENV = "MMM_QWEN35_MTP_ACTIVE_TUNING"
_DEFAULT_MTP_WIDTHS = "1,2,3,4,5,6,8"
_ALLOWED_DRAFT_KV = ("f16", "q8_0", "q4_0")
_ACTIVE_RUNTIME_KEYS = (
    "MMM_LLAMA_ACTIVE_SPEC_TYPE",
    "MMM_LLAMA_ACTIVE_DRAFT_N_MAX",
    "MMM_LLAMA_ACTIVE_PARALLEL",
    "MMM_LLAMA_ACTIVE_UBATCH",
    "MMM_LLAMA_ACTIVE_CACHE_REUSE",
    "MMM_LLAMA_ACTIVE_MTP_P_MIN",
    "MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE",
    "MMM_LLAMA_ACTIVE_KV_CACHE",
    "MMM_LLAMA_ACTIVE_DRAFT_KV_CACHE",
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
    """Return the authoritative llama.cpp context window for this model profile."""

    raw = os.environ.get("MMM_QWEN35_MTP_CTX", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("MMM_QWEN35_MTP_CTX must be a non-negative integer") from exc
        if value < 0:
            raise ValueError("MMM_QWEN35_MTP_CTX must be a non-negative integer")
        return min(_MAX_CTX, value)

    configured = getattr(config, "max_context", 0) if config is not None else 0
    try:
        value = int(configured)
    except (TypeError, ValueError):
        value = 0
    if value < 0:
        value = 0
    return min(_MAX_CTX, value)


def _draft_gpu_layers() -> str:
    """Return the Qwen MTP draft offload policy used by tuning and launch."""

    raw = os.environ.get("MMM_QWEN35_MTP_DRAFT_NGL", "all").strip().lower() or "all"
    if raw in {"all", "auto"}:
        return raw
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MMM_QWEN35_MTP_DRAFT_NGL must be 'all', 'auto', or a non-negative integer"
        ) from exc
    if value < 0:
        raise ValueError(
            "MMM_QWEN35_MTP_DRAFT_NGL must be 'all', 'auto', or a non-negative integer"
        )
    return str(value)


def _draft_kv_candidates() -> tuple[str, ...]:
    """Return draft-cache candidates; an explicit value disables the search."""

    explicit = os.environ.get("MMM_QWEN35_MTP_DRAFT_KV", "").strip().lower()
    if explicit:
        if explicit not in _ALLOWED_DRAFT_KV:
            raise ValueError(
                "MMM_QWEN35_MTP_DRAFT_KV must be one of f16, q8_0, q4_0"
            )
        return (explicit,)

    values: list[str] = []
    for token in os.environ.get(
        "MMM_QWEN35_MTP_DRAFT_KV_CANDIDATES", "f16,q8_0,q4_0"
    ).split(","):
        value = token.strip().lower()
        if value in _ALLOWED_DRAFT_KV and value not in values:
            values.append(value)
    return tuple(values or ("f16",))


def _draft_kv_from_variant(variant: Any) -> str:
    name = str(getattr(variant, "name", ""))
    marker = "|dkv-"
    if marker in name:
        value = name.split(marker, 1)[1].split("|", 1)[0].strip().lower()
        if value in _ALLOWED_DRAFT_KV:
            return value
    return _draft_kv_candidates()[0]


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


def _disable_decode_slot_polling() -> None:
    """Keep progress reporting local while Qwen is actively decoding on the T4."""

    from . import llama_server_hardware_policy as hardware_policy

    current = getattr(hardware_policy, "_slot_snapshot", None)
    if not callable(current) or getattr(current, _SLOT_POLL_MARKER, False):
        return

    @wraps(current)
    def no_decode_slot_poll(*_args: Any, **_kwargs: Any) -> None:
        return None

    setattr(no_decode_slot_poll, _SLOT_POLL_MARKER, True)
    hardware_policy._slot_snapshot = no_decode_slot_poll


def _install_measured_fast_base_args(autotune: Any) -> None:
    """Install the Qwen3.5 T4 launch constraints without masking decode tuners."""

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
        _set_option(args, ("--parallel", "-np"), "1")
        _set_option(args, ("--ctx-size", "-c"), str(_context_size(config)))

        # Main KV cache options are owned by the decode-speed autotuner.
        _drop_option(args, ("--load-mode", "-lm"))
        _drop_option(args, ("--cache-prompt",), takes_value=False)
        # Do not issue diagnostic HTTP requests to /slots during active decoding.
        _drop_option(args, ("--slots",), takes_value=False)
        if "--metrics" not in args:
            args.append("--metrics")
        return args

    setattr(measured_base_args, _BASE_MARKER, True)
    autotune._base_args = measured_base_args


def _install_measured_fast_variant_args(autotune: Any) -> None:
    """Keep Qwen MTP draft compute and its selected KV cache on the GPU hot path."""

    current = getattr(autotune, "_variant_args", None)
    if not callable(current) or getattr(current, _VARIANT_MARKER, False):
        return

    @wraps(current)
    def measured_variant_args(variant: Any) -> list[str]:
        args = list(current(variant))
        if (
            os.environ.get(_ACTIVE_TUNING_ENV, "").strip() == "1"
            and str(getattr(variant, "spec_type", "")) == "draft-mtp"
        ):
            _set_option(
                args,
                (
                    "--spec-draft-ngl",
                    "-ngld",
                    "--gpu-layers-draft",
                    "--n-gpu-layers-draft",
                ),
                _draft_gpu_layers(),
            )
            draft_kv = _draft_kv_from_variant(variant)
            _set_option(
                args,
                ("--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"),
                draft_kv,
            )
            _set_option(
                args,
                ("--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"),
                draft_kv,
            )
        return args

    setattr(measured_variant_args, _VARIANT_MARKER, True)
    autotune._variant_args = measured_variant_args


def _install_qwen35_fingerprint(autotune: Any) -> None:
    """Invalidate cached winners when Qwen draft memory policy changes."""

    current = getattr(autotune, "_fingerprint", None)
    if not callable(current) or getattr(current, _FINGERPRINT_MARKER, False):
        return

    @wraps(current)
    def qwen35_fingerprint(config: Any, binary: str, model_path: str) -> str:
        base = str(current(config, binary, model_path))
        if not _enabled() or not _is_qwen35_mtp(config):
            return base
        payload = {
            "base": base,
            "qwen35_mtp_draft_ngl": _draft_gpu_layers(),
            "qwen35_mtp_draft_kv_candidates": list(_draft_kv_candidates()),
            "qwen35_hotpath": "v11",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    setattr(qwen35_fingerprint, _FINGERPRINT_MARKER, True)
    autotune._fingerprint = qwen35_fingerprint


def _install_draft_kv_benchmark(autotune: Any) -> None:
    """Measure draft KV only after native MTP has already won the decode search."""

    current = getattr(autotune, "_benchmark", None)
    if not callable(current) or getattr(current, _DRAFT_KV_BENCHMARK_MARKER, False):
        return

    @wraps(current)
    def benchmark(
        binary: str,
        model_path: str,
        config: Any,
        request: Any,
        fingerprint: str,
    ) -> Any:
        decision = current(binary, model_path, config, request, fingerprint)
        if decision is None or not _enabled() or not _is_qwen35_mtp(config):
            return decision
        selected = getattr(decision, "selected", None)
        if selected is None or str(getattr(selected, "spec_type", "")) != "draft-mtp":
            return decision

        candidates = _draft_kv_candidates()
        run_variant = getattr(autotune, "_mmm_run_tuning_variant", None)
        if len(candidates) <= 1 or not callable(run_variant):
            return decision

        benchmark_request = autotune._compact_benchmark_request(request)
        probe_tokens = min(
            int(config.max_new_tokens),
            autotune._env_int(
                "MMM_LLAMA_AUTOTUNE_TOKENS",
                autotune._BENCHMARK_OUTPUT_TOKENS,
            ),
        )
        base_name = str(getattr(selected, "name", "mtp")).split("|dkv-", 1)[0]
        probes: list[Any] = []
        for kv in candidates:
            variant = replace(selected, name=f"{base_name}|dkv-{kv}")
            probe = run_variant(
                binary,
                model_path,
                config,
                benchmark_request,
                variant,
                probe_tokens=probe_tokens,
            )
            probes.append(probe)

        reference = next(
            (
                probe
                for probe in probes
                if bool(getattr(probe, "ok", False))
                and float(getattr(probe, "predicted_tps", 0.0)) > 0
            ),
            None,
        )
        if reference is None:
            return decision
        reference_hash = str(getattr(reference, "output_sha256", ""))
        eligible = [
            probe
            for probe in probes
            if bool(getattr(probe, "ok", False))
            and float(getattr(probe, "predicted_tps", 0.0)) > 0
            and str(getattr(probe, "output_sha256", "")) == reference_hash
        ]
        if not eligible:
            return decision

        best = max(eligible, key=lambda probe: float(getattr(probe, "predicted_tps", 0.0)))
        minimum_gain = max(
            1.0,
            autotune._env_float("MMM_LLAMA_STAGE_MIN_GAIN", 1.01),
        )
        reference_tps = max(1e-9, float(getattr(reference, "predicted_tps", 0.0)))
        if (
            best is not reference
            and float(getattr(best, "predicted_tps", 0.0)) / reference_tps < minimum_gain
        ):
            best = reference

        selected_tps = float(getattr(best, "predicted_tps", 0.0))
        baseline_tps = float(getattr(decision, "baseline_tps", 0.0))
        return autotune.AutotuneDecision(
            fingerprint=fingerprint,
            selected=best.variant,
            baseline_tps=baseline_tps,
            selected_tps=selected_tps,
            speedup=(selected_tps / baseline_tps if baseline_tps > 0 else 1.0),
            probes=tuple(getattr(decision, "probes", ())) + tuple(probes),
        )

    setattr(benchmark, _DRAFT_KV_BENCHMARK_MARKER, True)
    autotune._benchmark = benchmark


def _install_draft_kv_launch_export(autotune: Any) -> None:
    current = getattr(autotune, "_launch_selected", None)
    if not callable(current) or getattr(current, _DRAFT_KV_LAUNCH_MARKER, False):
        return

    @wraps(current)
    def launch_selected(binary: str, model_path: str, config: Any, selected: Any) -> str:
        url = current(binary, model_path, config, selected)
        if _is_qwen35_mtp(config) and str(getattr(selected, "spec_type", "")) == "draft-mtp":
            os.environ["MMM_LLAMA_ACTIVE_DRAFT_KV_CACHE"] = _draft_kv_from_variant(selected)
        else:
            os.environ.pop("MMM_LLAMA_ACTIVE_DRAFT_KV_CACHE", None)
        return url

    setattr(launch_selected, _DRAFT_KV_LAUNCH_MARKER, True)
    autotune._launch_selected = launch_selected


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


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def install(autotune: Any) -> None:
    """Keep Qwen3.5 on the T4 hot path while delegating winner selection."""

    _disable_decode_slot_polling()
    _install_measured_fast_base_args(autotune)
    _install_measured_fast_variant_args(autotune)
    _install_qwen35_fingerprint(autotune)
    _install_draft_kv_benchmark(autotune)
    _install_draft_kv_launch_export(autotune)
    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_qwen35_measured(config: Any, request: Any) -> str:
        if not _enabled() or not _is_qwen35_mtp(config):
            return current(config, request)

        managed_process = getattr(autotune, "_MANAGED_PROCESS", None)
        managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")
        if managed_process is not None and managed_process.poll() is None and managed_url:
            return current(config, request)

        _reclaim_prior_mmm_server()
        previous_ctx = os.environ.get("MMM_LLAMA_SERVER_CTX")
        previous_widths = os.environ.get("MMM_LLAMA_MTP_WIDTHS")
        previous_active = os.environ.get(_ACTIVE_TUNING_ENV)
        os.environ["MMM_LLAMA_SERVER_CTX"] = str(_context_size(config))
        os.environ[_ACTIVE_TUNING_ENV] = "1"
        if not (previous_widths or "").strip():
            os.environ["MMM_LLAMA_MTP_WIDTHS"] = _DEFAULT_MTP_WIDTHS
        try:
            return current(config, request)
        finally:
            _restore_env("MMM_LLAMA_SERVER_CTX", previous_ctx)
            _restore_env("MMM_LLAMA_MTP_WIDTHS", previous_widths)
            _restore_env(_ACTIVE_TUNING_ENV, previous_active)

    setattr(ensure_qwen35_measured, _MARKER, True)
    ensure_qwen35_measured._mmm_qwen35_measured_decode_hotpath = True  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_qwen35_measured


__all__ = [
    "_context_size",
    "_disable_decode_slot_polling",
    "_draft_gpu_layers",
    "_draft_kv_candidates",
    "_draft_kv_from_variant",
    "_install_draft_kv_benchmark",
    "_install_measured_fast_base_args",
    "_install_measured_fast_variant_args",
    "_is_qwen35_mtp",
    "_prior_mmm_loopback_port",
    "install",
]
