"""Commercial-permissive foreground alpha for opaque diffusion image assets.

FLUX.2 Klein 4B's local Diffusers output is RGB, not native transparent PNG.
The host requests a *specific* BiRefNet general segmentation checkpoint via
rembg (MIT). NEVER let rembg choose its default: its default Bria weights have
a separate non-commercial license.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ALPHA_SEGMENTATION_MODEL = "birefnet-general"
ALPHA_SEGMENTATION_ENGINE = "rembg"
ALPHA_SEGMENTATION_PROVIDER = "CPUExecutionProvider"
ALPHA_SEGMENTATION_CONTRACT = "mmm/alpha-segmentation-birefnet-general-v1"


def _require_custom_session_options_api(new_session: Any) -> None:
    """Reject rembg before model download if it cannot accept ONNX session opts.

    rembg <=2.0.76 always constructs its own sess_opts and forwards an explicitly
    supplied sess_opts again via **kwargs, crashing BaseSession.__init__. Silently
    dropping these settings is unsafe for BiRefNet under Colab RAM pressure.
    """
    import inspect

    try:
        parameter = inspect.signature(new_session).parameters.get("sess_opts")
    except (ValueError, TypeError):
        parameter = None
    if parameter is None or parameter.kind not in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("rembg")
        except PackageNotFoundError:
            installed = "not-installed"
        raise ValueError(
            "ALPHA_SEGMENTER_INCOMPATIBLE_REMBG_API: "
            f"installed rembg={installed}; new_session(sess_opts=...) unavailable. "
            "Rerun Colab setup cell 2 to install rembg[cpu]>=2.0.77,<3 "
            "before image generation."
        )


@lru_cache(maxsize=1)
def _get_session() -> Any:
    try:
        from rembg import new_session
    except ImportError as exc:
        raise ValueError(
            'ALPHA_SEGMENTER_UNAVAILABLE: install "rembg[cpu]" via the image extra; '
            "opaque diffusion sprites cannot be published without alpha segmentation."
        ) from exc
    _require_custom_session_options_api(new_session)
    # BiRefNet's full-size session can exceed Colab host RAM when FLUX and a
    # JVM have recently been active. Constrain ONNX allocations at creation
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

    # Explicit model ID is a licensing contract. rembg's DEFAULT is not allowed.
    return new_session(
        ALPHA_SEGMENTATION_MODEL,
        sess_opts=session_options,
        providers=[ALPHA_SEGMENTATION_PROVIDER],
    )


def segment_foreground(image: Any) -> Any:
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
        result = remove(source, session=_get_session(), alpha_matting=False)
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
    result.info["mmm_alpha_matte"] = ALPHA_SEGMENTATION_ENGINE + ":" + ALPHA_SEGMENTATION_MODEL
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


def _preflight_worker_ram() -> None:
    """Fail explicitly rather than asking the OOM killer to choose a process."""
    raw = os.environ.get("MMM_ALPHA_MIN_AVAILABLE_MB", "3072")
    try:
        minimum_mb = int(raw)
    except ValueError as exc:
        raise ValueError("MMM_ALPHA_MIN_AVAILABLE_MB must be an integer") from exc
    if minimum_mb < 0:
        raise ValueError("MMM_ALPHA_MIN_AVAILABLE_MB cannot be negative")
    available = _available_host_ram_bytes()
    if available is not None and available < minimum_mb * 1024 * 1024:
        raise ValueError(
            "ALPHA_SEGMENTER_INSUFFICIENT_HOST_RAM: "
            f"{available // (1024 * 1024)} MiB free; "
            f"requires at least {minimum_mb} MiB before starting BiRefNet. "
            "Free host RAM, stop concurrent JVM builds, or increase Colab RAM."
        )


def _preflight_checkpoint_download_ram() -> None:
    """A streaming checkpoint fetch is not an ONNX inference allocation.

    The full BiRefNet inference worker retains its stricter, separate 3 GiB
    safety gate. Checkpoint preparation only needs room for the Python/pooch
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
    """Download the checksummed BiRefNet checkpoint before FLUX is resident.

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

    _preflight_worker_ram()
    with tempfile.TemporaryDirectory(prefix="mmm-alpha-") as temporary:
        source = Path(temporary) / "input.png"
        target = Path(temporary) / "output.png"
        image.convert("RGB").save(source, format="PNG")
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

        _used_before, _limit_before, memory_events_before = _cgroup_memory()
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
                f"ALPHA_SEGMENTER_WORKER_TIMEOUT: BiRefNet exceeded {timeout}s"
            ) from exc
        if completed.returncode != 0:
            # The kernel's OOM counter separates a confirmed cgroup OOM kill
            # from a SIGKILL sent by some other process. Keep the original
            # failure class for callers that must fail closed.
            _used_after, _limit_after, memory_events_after = _cgroup_memory()
            oom_kill_delta = max(
                0,
                memory_events_after.get("oom_kill", 0)
                - memory_events_before.get("oom_kill", 0),
            )
            memory_headroom = _available_host_ram_bytes()
            # Progress bars fill stderr with thousands of redraws; report
            # diagnostics, not repeated download percentages.
            stderr = completed.stderr or ""
            stderr_lines = stderr.replace("\r", "\n").splitlines()
            # Keep enough traceback context instead of just its final frames.
            tail = " | ".join(stderr_lines[-20:])[-3000:]
            incompatible_api = (
                "ALPHA_SEGMENTER_INCOMPATIBLE_REMBG_API" in stderr
                or "multiple values for argument 'sess_opts'" in stderr
            )
            possible_oom = completed.returncode in (-9, 137)
            if incompatible_api:
                kind = "ALPHA_SEGMENTER_INCOMPATIBLE_REMBG_API"
            elif possible_oom:
                kind = "ALPHA_SEGMENTER_WORKER_OOM_SUSPECT"
            else:
                kind = "ALPHA_SEGMENTER_WORKER_FAILED"
            remediation = (
                " Rerun Colab setup cell 2 to upgrade rembg[cpu]>=2.0.77,<3."
                if incompatible_api else ""
            )
            raise ValueError(
                f"{kind}: exit={completed.returncode}; "
                f"cgroup_oom_kill_delta={oom_kill_delta}; "
                f"available_mib={memory_headroom // 1048576 if memory_headroom is not None else 'unknown'}; "
                f"stderr_tail={tail}{remediation}"
            )
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
        detached.info["mmm_alpha_matte"] = ALPHA_SEGMENTATION_ENGINE + ":" + ALPHA_SEGMENTATION_MODEL
        return detached


def _worker_main(source: str, target: str) -> None:
    """Subprocess-only entry point; the parent never imports ONNX Runtime."""
    from PIL import Image

    with Image.open(source) as raw:
        raw.load()
        result = segment_foreground(raw)
        try:
            result.save(target, format="PNG", optimize=False)
        finally:
            result.close()


__all__ = [
    "ALPHA_SEGMENTATION_MODEL",
    "ALPHA_SEGMENTATION_ENGINE",
    "ALPHA_SEGMENTATION_PROVIDER",
    "ALPHA_SEGMENTATION_CONTRACT",
    "segment_foreground",
    "segment_foreground_isolated",
    "prepare_foreground_model_isolated",
]


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--prepare":
        from rembg import new_session
        from rembg.sessions.birefnet_general import BiRefNetSessionGeneral

        # Validate the worker API before the 1+ GiB checkpoint download and
        # before the diffusion model is loaded by the next generation step.
        _require_custom_session_options_api(new_session)
        # This is rembg's checksummed model downloader, not inference.
        BiRefNetSessionGeneral.download_models()
    elif len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(
            "Usage: python -m minecraft_mod_ai.resource_alpha_segmentation "
            "--prepare | --worker INPUT OUTPUT"
        )
