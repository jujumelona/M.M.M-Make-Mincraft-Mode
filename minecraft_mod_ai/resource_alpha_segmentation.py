"""Commercial-permissive foreground alpha for opaque diffusion image assets.

FLUX.2 Klein 4B's local Diffusers output is RGB, not native transparent PNG.
Use explicitly licensed BiRefNet-General-Lite (MIT), full U2Net
(Apache-2.0) for constrained hosts, and U2NetP (Apache-2.0) only for
low-memory/OOM fallback. Never use rembg's non-commercial BRIA default.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ALPHA_SEGMENTATION_MODEL = "birefnet-general-lite"
ALPHA_SEGMENTATION_MID_MODEL = "u2net"
ALPHA_SEGMENTATION_FALLBACK_MODEL = "u2netp"
ALPHA_SEGMENTATION_ENGINE = "rembg"
ALPHA_SEGMENTATION_PROVIDER = "CPUExecutionProvider"
ALPHA_SEGMENTATION_CONTRACT = "mmm/alpha-segmentation-birefnet-lite-u2net-u2netp-v3"
# Once the primary was SIGKILLed, skip further risky ONNX attempts in this
# Python kernel. The flag is not persisted between Colab sessions.
_LITE_WORKER_SIGKILLED = False
_U2NET_WORKER_SIGKILLED = False



@lru_cache(maxsize=1)
def _get_session(model_name: str = ALPHA_SEGMENTATION_MODEL) -> Any:
    try:
        if model_name == ALPHA_SEGMENTATION_MODEL:
            from rembg.sessions.birefnet_general_lite import BiRefNetSessionGeneralLite as model_class
        elif model_name == ALPHA_SEGMENTATION_MID_MODEL:
            from rembg.sessions.u2net import U2netSession as model_class
        elif model_name == ALPHA_SEGMENTATION_FALLBACK_MODEL:
            from rembg.sessions.u2netp import U2netpSession as model_class
        else:
            raise ValueError(f"ALPHA_SEGMENTER_MODEL_NOT_ALLOWED: {model_name}")
    except ImportError as exc:
        raise ValueError(
            'ALPHA_SEGMENTER_UNAVAILABLE: install "rembg[cpu]==2.0.67" via the '
            'image extra; BiRefNet general and explicit ONNX options are required.'
        ) from exc
    # Even the lightweight model can exceed constrained Colab host RAM.
    # Constrain ONNX allocations at creation
    # rather than only setting an OpenMP hint on its parent process.
    import onnxruntime as ort

    threads_raw = os.environ.get("MMM_ALPHA_ONNX_THREADS", "1")
    try:
        threads = int(threads_raw)
    except ValueError as exc:
        raise ValueError("MMM_ALPHA_ONNX_THREADS must be an integer") from exc
    if not 1 <= threads <= 4:
        raise ValueError("MMM_ALPHA_ONNX_THREADS must be between 1 and 4")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False

    # rembg 2.0.67's public new_session() *always* constructs sess_opts and
    # double-passes user supplied sess_opts. Construct its exported concrete
    # BiRefNet session instead; BaseSession(model, sess_opts, providers=...)
    # preserves all ONNX memory controls and the explicit licensed model choice.
    if model_class.name() != model_name:
        raise ValueError("ALPHA_SEGMENTER_MODEL_CLASS_MISMATCH")
    return model_class(
        model_name,
        session_options,
        providers=[ALPHA_SEGMENTATION_PROVIDER],
    )


def segment_foreground(image: Any, *, model_name: str = ALPHA_SEGMENTATION_MODEL) -> Any:
    """Return a separate RGBA image with model-produced foreground alpha.

    Never fake transparency by cutting a geometry-shaped region, and never
    accept a grayscale/RGB image returned by a misconfigured backend.
    Run ONNX on CPU so a resident 4B diffusion model does not OOM Colab T4.
    """
    from PIL import Image

    try:
        from rembg import remove
    except ImportError as exc:
        raise ValueError(
            'ALPHA_SEGMENTER_UNAVAILABLE: install "rembg[cpu]" via the image extra.'
        ) from exc

    source = image.convert("RGB")
    try:
        result = remove(source, session=_get_session(model_name), alpha_matting=False)
    finally:
        source.close()
    if not isinstance(result, Image.Image) or result.size != image.size:
        raise ValueError("ALPHA_SEGMENTER_INVALID_IMAGE_OR_GEOMETRY")
    if result.mode != "RGBA":
        result.close()
        raise ValueError("ALPHA_SEGMENTER_MISSING_RGBA_ALPHA")
    alpha = result.getchannel("A")
    try:
        low, high = alpha.getextrema()
        if low == high or high == 0:
            result.close()
            raise ValueError("ALPHA_SEGMENTER_FAILED_TO_EXTRACT_SUBJECT")
    finally:
        alpha.close()
    result.info["mmm_alpha_matte"] = ALPHA_SEGMENTATION_ENGINE + ":" + model_name
    return result


def _available_host_ram_bytes() -> int | None:
    """Return the effective RAM headroom, accounting for container limits."""
    from .runtime_memory_watchdog import _cgroup_memory, _meminfo

    _total, available = _meminfo()
    used, limit, _events = _cgroup_memory()
    if limit > 0:
        remaining = max(0, limit - used)
        return min(available, remaining) if available else remaining
    return available or None


def _preflight_worker_ram(*, model_name: str = ALPHA_SEGMENTATION_MODEL) -> None:
    """Check the selected worker's RAM floor, not the largest model's floor."""
    if model_name not in {ALPHA_SEGMENTATION_MODEL, ALPHA_SEGMENTATION_MID_MODEL, ALPHA_SEGMENTATION_FALLBACK_MODEL}:
        raise ValueError(f"ALPHA_SEGMENTER_MODEL_NOT_ALLOWED: {model_name}")
    # Keep the full-model gate intact; U2NetP has a smaller ONNX footprint.
    config_key = ("MMM_ALPHA_U2NETP_MIN_AVAILABLE_MB"
                  if model_name == ALPHA_SEGMENTATION_FALLBACK_MODEL
                  else "MMM_ALPHA_MIN_AVAILABLE_MB")
    raw = (
        os.environ.get("MMM_ALPHA_U2NETP_MIN_AVAILABLE_MB", "2048")
        if model_name == ALPHA_SEGMENTATION_FALLBACK_MODEL
        else os.environ.get("MMM_ALPHA_MIN_AVAILABLE_MB", "3072")
    )
    try:
        minimum_mb = int(raw)
    except ValueError as exc:
        raise ValueError(f"{config_key} must be an integer") from exc
    if minimum_mb < 0:
        raise ValueError(f"{config_key} cannot be negative")
    available = _available_host_ram_bytes()
    if available is not None and available < minimum_mb * 1024 * 1024:
        raise ValueError(
            "ALPHA_SEGMENTER_INSUFFICIENT_HOST_RAM: "
            f"{available // (1024 * 1024)} MiB free; "
            f"requires at least {minimum_mb} MiB before starting {model_name}. "
            "Free host RAM, stop concurrent JVM builds, or increase Colab RAM."
        )


def _preflight_checkpoint_download_ram() -> None:
    """A streaming checkpoint fetch is not an ONNX inference allocation.

    The ONNX inference worker has a separate model-specific RAM gate
    (3 GiB for full models; 2 GiB for U2NetP). Checkpoint preparation only needs room for the Python/pooch
    download subprocess and disk-backed streamed bytes.
    """
    raw = os.environ.get("MMM_ALPHA_PREPARE_MIN_AVAILABLE_MB", "512")
    try:
        minimum_mb = int(raw)
    except ValueError as exc:
        raise ValueError("MMM_ALPHA_PREPARE_MIN_AVAILABLE_MB must be an integer") from exc
    if minimum_mb < 0:
        raise ValueError("MMM_ALPHA_PREPARE_MIN_AVAILABLE_MB cannot be negative")
    available = _available_host_ram_bytes()
    if available is not None and available < minimum_mb * 1048576:
        raise ValueError(
            "ALPHA_SEGMENTER_PREPARE_INSUFFICIENT_HOST_RAM: "
            f"{available // 1048576} MiB free, "
            f"requires {minimum_mb} MiB before a streaming checkpoint download"
        )


def prepare_foreground_model_isolated() -> None:
    """Download both checksummed foreground models before FLUX is resident.

    A short-lived subprocess avoids overlapping first-time model downloads
    with the large CPU-offloaded diffusion pipeline.
    """
    timeout_raw = os.environ.get("MMM_ALPHA_PREPARE_TIMEOUT_SECONDS", "480")
    try:
        timeout = int(timeout_raw)
    except ValueError as exc:
        raise ValueError("MMM_ALPHA_PREPARE_TIMEOUT_SECONDS must be an integer") from exc
    if timeout < 1:
        raise ValueError("MMM_ALPHA_PREPARE_TIMEOUT_SECONDS must be positive")
    _preflight_checkpoint_download_ram()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "minecraft_mod_ai.resource_alpha_segmentation",
             "--prepare"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ALPHA_SEGMENTER_PREPARE_TIMEOUT") from exc
    if completed.returncode:
        raise ValueError(
            f"ALPHA_SEGMENTER_PREPARE_FAILED: exit={completed.returncode}; "
            + (completed.stderr or "")[-900:]
        )


def segment_foreground_isolated(image: Any) -> Any:
    """Run CPU ONNX inference outside the FLUX-owning Python process.

    Native ONNX crashes become explicit subprocess errors, and allocations are
    returned to the OS when the worker exits. Production must call this only
    after releasing the diffusion pipeline's CPU-offloaded weights.
    """
    from PIL import Image

    global _LITE_WORKER_SIGKILLED, _U2NET_WORKER_SIGKILLED
    # BiRefNet-Lite exceeded the observed ~6 GiB Colab memory headroom.
    # Full U2Net is materially more capable than the tiny U2NetP and fits
    # intermediate memory budgets. Do not silently downgrade a 5-7 GiB
    # machine to the smallest segmenter.
    headroom = _available_host_ram_bytes()
    minimum_lite_mib = int(os.environ.get("MMM_ALPHA_LITE_MIN_AVAILABLE_MIB", "8192"))
    minimum_u2net_mib = int(os.environ.get("MMM_ALPHA_U2NET_MIN_AVAILABLE_MIB", "4608"))
    if minimum_lite_mib < 3072:
        raise ValueError("MMM_ALPHA_LITE_MIN_AVAILABLE_MIB must be >= 3072")
    if not 3072 <= minimum_u2net_mib <= minimum_lite_mib:
        raise ValueError("MMM_ALPHA_U2NET_MIN_AVAILABLE_MIB must be 3072..MMM_ALPHA_LITE_MIN_AVAILABLE_MIB")
    if _LITE_WORKER_SIGKILLED or _U2NET_WORKER_SIGKILLED:
        model_order = (ALPHA_SEGMENTATION_FALLBACK_MODEL,)
    elif headroom is None or headroom >= minimum_lite_mib * 1048576:
        model_order = (ALPHA_SEGMENTATION_MODEL, ALPHA_SEGMENTATION_FALLBACK_MODEL)
    elif headroom >= minimum_u2net_mib * 1048576:
        model_order = (ALPHA_SEGMENTATION_MID_MODEL, ALPHA_SEGMENTATION_FALLBACK_MODEL)
    else:
        model_order = (ALPHA_SEGMENTATION_FALLBACK_MODEL,)
    if model_order[0] != ALPHA_SEGMENTATION_MODEL:
        print(
            f"ALPHA_SEGMENTER_MEMORY_POLICY: selected={model_order[0]} "
            f"available_mib={headroom // 1048576 if headroom is not None else 'unknown'} "
            f"lite_min_mib={minimum_lite_mib} u2net_min_mib={minimum_u2net_mib} "
            f"previous_lite_kill={_LITE_WORKER_SIGKILLED} "
            f"previous_u2net_kill={_U2NET_WORKER_SIGKILLED}",
            flush=True,
        )
    with tempfile.TemporaryDirectory(prefix="mmm-alpha-") as temporary:
        source = Path(temporary) / "input.png"
        target = Path(temporary) / "output.png"
        with image.convert("RGB") as converted:
            converted.save(source, format="PNG")
        timeout_raw = os.environ.get("MMM_ALPHA_WORKER_TIMEOUT_SECONDS", "300")
        try:
            timeout = max(1, int(timeout_raw))
        except ValueError as exc:
            raise ValueError("MMM_ALPHA_WORKER_TIMEOUT_SECONDS must be an integer") from exc
        env = os.environ.copy()
        # A large global OMP setting must not override the isolated worker's
        # low-memory ONNX session policy.
        env["OMP_NUM_THREADS"] = "1"
        env["OMP_THREAD_LIMIT"] = "1"
        from .runtime_memory_watchdog import _cgroup_memory

        # Isolated licensed workers avoid simultaneous FLUX/ONNX residency.
        # OOM and semantically unusable masks can try the next explicit model;
        # API and other infrastructure errors remain fail-closed.
        failures = []
        chosen_model = None
        for model_name in model_order:
            # Recheck immediately before launch: JVM/FLUX allocations can
            # consume RAM after model selection. A larger model cannot block
            # the explicitly permitted U2NetP recovery path.
            try:
                _preflight_worker_ram(model_name=model_name)
            except ValueError as exc:
                if (
                    str(exc).startswith("ALPHA_SEGMENTER_INSUFFICIENT_HOST_RAM:")
                    and model_name != model_order[-1]
                ):
                    failures.append(str(exc))
                    print(f"ALPHA_SEGMENTER_MEMORY_RECOVERY: skipping={model_name}; "
                          f"trying={model_order[model_order.index(model_name) + 1]}; {exc}", flush=True)
                    continue
                raise
            target.unlink(missing_ok=True)  # Never reuse a prior worker's mask.
            _used_before, _limit_before, memory_events_before = _cgroup_memory()
            env["MMM_ALPHA_WORKER_MODEL"] = model_name
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "minecraft_mod_ai.resource_alpha_segmentation",
                     "--worker", str(source), str(target)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError(
                    f"ALPHA_SEGMENTER_WORKER_TIMEOUT: model={model_name}; worker exceeded {timeout}s"
                ) from exc
            if completed.returncode == 0:
                if not target.is_file() or target.is_symlink():
                    raise ValueError("ALPHA_SEGMENTER_WORKER_MISSING_OUTPUT")
                with Image.open(target) as candidate:
                    candidate.load()
                    if candidate.mode != "RGBA" or candidate.size != image.size:
                        raise ValueError("ALPHA_SEGMENTER_INVALID_IMAGE_OR_GEOMETRY")
                    with candidate.getchannel("A") as alpha:
                        histogram = alpha.histogram()
                    visible_fraction = sum(histogram[128:]) / (image.width * image.height)
                if 0 < visible_fraction < 0.98:
                    chosen_model = model_name
                    break
                failures.append(
                    f"ALPHA_SEGMENTER_UNUSABLE_MASK: model={model_name}; "
                    f"native_threshold_coverage={visible_fraction:.6f}"
                )
                if model_name == model_order[-1]:
                    raise ValueError(" | ".join(failures))
                print(
                    f"ALPHA_SEGMENTER_MASK_RECOVERY: model={model_name} "
                    f"coverage={visible_fraction:.6f}; "
                    f"trying={model_order[model_order.index(model_name) + 1]}",
                    flush=True,
                )
                continue

            _used_after, _limit_after, memory_events_after = _cgroup_memory()
            oom_kill_delta = max(
                0, memory_events_after.get("oom_kill", 0)
                - memory_events_before.get("oom_kill", 0),
            )
            memory_headroom = _available_host_ram_bytes()
            stderr = completed.stderr or ""
            stderr_lines = stderr.replace("\r", "\n").splitlines()
            tail = " | ".join(stderr_lines[-20:])[-3000:]
            incompatible_api = (
                "ALPHA_SEGMENTER_INCOMPATIBLE_REMBG_API" in stderr
                or "multiple values for argument 'sess_opts'" in stderr
            )
            possible_oom = completed.returncode in (-9, 137)
            semantic_failure = "ALPHA_SEGMENTER_FAILED_TO_EXTRACT_SUBJECT" in stderr
            if possible_oom and model_name == ALPHA_SEGMENTATION_MODEL:
                _LITE_WORKER_SIGKILLED = True
            if possible_oom and model_name == ALPHA_SEGMENTATION_MID_MODEL:
                _U2NET_WORKER_SIGKILLED = True
            if incompatible_api:
                kind = "ALPHA_SEGMENTER_INCOMPATIBLE_REMBG_API"
            elif possible_oom:
                kind = "ALPHA_SEGMENTER_WORKER_OOM_SUSPECT"
            elif semantic_failure:
                kind = "ALPHA_SEGMENTER_UNUSABLE_MASK"
            else:
                kind = "ALPHA_SEGMENTER_WORKER_FAILED"
            remediation = (
                " Rerun Colab setup cell 2 to install rembg[cpu]==2.0.67."
                if incompatible_api else ""
            )
            failures.append(
                f"{kind}: model={model_name}; exit={completed.returncode}; "
                f"cgroup_oom_kill_delta={oom_kill_delta}; "
                f"available_mib={memory_headroom // 1048576 if memory_headroom is not None else 'unknown'}; "
                f"stderr_tail={tail}{remediation}"
            )
            # SIGKILL alone is not proof of cgroup OOM when delta is zero.
            # Retry only resource exhaustion or a *recognized mask-quality*
            # error. A broken ONNX/backend must not be silently concealed.
            if (not possible_oom and not semantic_failure) or model_name == model_order[-1]:
                raise ValueError(" | ".join(failures))
            print(
                f"ALPHA_SEGMENTER_RECOVERY: worker={model_name}; "
                f"trying fallback={model_order[model_order.index(model_name) + 1]}; "
                + failures[-1],
                flush=True,
            )
        if chosen_model is None:
            raise ValueError(" | ".join(failures))
        if not target.is_file():
            raise ValueError("ALPHA_SEGMENTER_WORKER_MISSING_OUTPUT")
        with Image.open(target) as result:
            result.load()
            if result.mode != "RGBA" or result.size != image.size:
                raise ValueError("ALPHA_SEGMENTER_INVALID_IMAGE_OR_GEOMETRY")
            alpha = result.getchannel("A")
            try:
                low, high = alpha.getextrema()
            finally:
                alpha.close()
            if low == high or high == 0:
                raise ValueError("ALPHA_SEGMENTER_FAILED_TO_EXTRACT_SUBJECT")
            detached = result.copy()
        detached.info["mmm_alpha_matte"] = ALPHA_SEGMENTATION_ENGINE + ":" + chosen_model
        return detached


def _worker_main(source: str, target: str) -> None:
    """Subprocess-only entry point; the parent never imports ONNX Runtime."""
    from PIL import Image

    with Image.open(source) as raw:
        raw.load()
        result = segment_foreground(raw, model_name=os.environ.get("MMM_ALPHA_WORKER_MODEL", ALPHA_SEGMENTATION_MODEL))
        try:
            result.save(target, format="PNG", optimize=False)
        finally:
            result.close()


__all__ = [
    "ALPHA_SEGMENTATION_MODEL",
    "ALPHA_SEGMENTATION_MID_MODEL",
    "ALPHA_SEGMENTATION_FALLBACK_MODEL",
    "ALPHA_SEGMENTATION_ENGINE",
    "ALPHA_SEGMENTATION_PROVIDER",
    "ALPHA_SEGMENTATION_CONTRACT",
    "segment_foreground",
    "segment_foreground_isolated",
    "prepare_foreground_model_isolated",
]


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--prepare":
        from rembg.sessions.birefnet_general_lite import BiRefNetSessionGeneralLite
        from rembg.sessions.u2net import U2netSession
        from rembg.sessions.u2netp import U2netpSession
        # Prefetch all explicitly permitted, checksum-verified ONNX weights.
        # Neither the noncommercial BRIA weights nor rembg's default is used.
        for model_class, expected in (
            (BiRefNetSessionGeneralLite, ALPHA_SEGMENTATION_MODEL),
            (U2netSession, ALPHA_SEGMENTATION_MID_MODEL),
            (U2netpSession, ALPHA_SEGMENTATION_FALLBACK_MODEL),
        ):
            if model_class.name() != expected:
                raise ValueError("ALPHA_SEGMENTER_MODEL_CLASS_MISMATCH")
            model_class.download_models()
    elif len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(
            "Usage: python -m minecraft_mod_ai.resource_alpha_segmentation "
            "--prepare | --worker INPUT OUTPUT"
        )
