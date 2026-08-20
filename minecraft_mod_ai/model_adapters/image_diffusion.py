from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .base import (
    ModelBackendError,
    ModelConfigurationError,
    _release_cuda,
    preflight_cuda,
    require_package,
    torch_dtype,
)


_IMAGE_LOCK = threading.RLock()
_IMAGE_PIPELINE: Any | None = None
_IMAGE_PIPELINE_KEY: tuple[str, str, str] | None = None
_IMAGE_PIPELINE_ON_GPU = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _full_gpu_threshold_mb(config: Any) -> int:
    raw = os.environ.get("MMM_IMAGE_FULL_GPU_MIN_FREE_MB", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return max(14_000, int(config.min_free_vram_mb) + 1_000)


def _is_cuda_memory_pressure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "out of memory",
            "cuda oom",
            "cudnn_status_alloc_failed",
            "cublas_status_alloc_failed",
            "allocation failed",
            "not enough memory",
        )
    )


def _clear_cached_pipeline() -> None:
    global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY, _IMAGE_PIPELINE_ON_GPU

    _IMAGE_PIPELINE = None
    _IMAGE_PIPELINE_KEY = None
    _IMAGE_PIPELINE_ON_GPU = False


def _load_pipeline(config: Any) -> Any:
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype(config.torch_dtype),
        trust_remote_code=False,
    )
    progress = getattr(pipeline, "set_progress_bar_config", None)
    if callable(progress):
        progress(disable=True)
    return pipeline


def _activate_cpu_offload(
    pipeline: Any,
    config: Any,
    *,
    cache_enabled: bool,
) -> tuple[str, tuple[str, str, str]]:
    global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY, _IMAGE_PIPELINE_ON_GPU

    try:
        pipeline.to("cpu")
    finally:
        _IMAGE_PIPELINE_ON_GPU = False
        _release_cuda()
    pipeline.enable_model_cpu_offload()
    residency = "cpu_offload"
    key = (config.model_id, config.torch_dtype, residency)
    if cache_enabled:
        _IMAGE_PIPELINE = pipeline
        _IMAGE_PIPELINE_KEY = key
    else:
        _clear_cached_pipeline()
    return residency, key


def finish_image_shard() -> None:
    """Park or release the process-scoped image pipeline after an asset shard."""

    global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY, _IMAGE_PIPELINE_ON_GPU

    keep_across_shards = _env_bool("MMM_IMAGE_CACHE_ACROSS_SHARDS", False)
    with _IMAGE_LOCK:
        pipeline = _IMAGE_PIPELINE
        key = _IMAGE_PIPELINE_KEY
        if pipeline is None:
            _release_cuda()
            return

        if keep_across_shards:
            if key is not None and key[-1] == "full_gpu" and _IMAGE_PIPELINE_ON_GPU:
                pipeline.to("cpu")
                _IMAGE_PIPELINE_ON_GPU = False
        else:
            _clear_cached_pipeline()
            del pipeline
    _release_cuda()


class ImageDiffusionAdapter:
    """Diffusion backend with native process-scoped pipeline residency."""

    def __init__(self, config) -> None:
        self.config = config

    def generate_image(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Path:
        global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY, _IMAGE_PIPELINE_ON_GPU

        cfg = self.config
        try:
            require_package("diffusers", minimum="0.20.0")
            require_package("transformers", minimum="4.52.0")
            require_package("accelerate", minimum="1.0.0")
            if not prompt.strip():
                raise ModelConfigurationError("Image prompt is empty.")
            if (
                width % 16
                or height % 16
                or not (256 <= width <= 1024 and 256 <= height <= 1024)
            ):
                raise ModelConfigurationError(
                    "Image dimensions must be 256-1024 and divisible by 16."
                )

            preflight_cuda(cfg)
            import torch

            free_bytes, _ = torch.cuda.mem_get_info()
            free_mb = int(free_bytes / (1024 * 1024))
            use_full_gpu = not cfg.cpu_offload or free_mb >= _full_gpu_threshold_mb(cfg)
            residency = "full_gpu" if use_full_gpu else "cpu_offload"
            cache_enabled = _env_bool("MMM_IMAGE_PIPELINE_CACHE", True)

            with _IMAGE_LOCK:
                key = (cfg.model_id, cfg.torch_dtype, residency)
                pipeline = (
                    _IMAGE_PIPELINE
                    if cache_enabled and _IMAGE_PIPELINE_KEY == key
                    else None
                )
                if pipeline is None:
                    if _IMAGE_PIPELINE is not None:
                        _clear_cached_pipeline()
                        _release_cuda()
                    pipeline = _load_pipeline(cfg)
                    if residency == "full_gpu":
                        try:
                            pipeline.to("cuda")
                            _IMAGE_PIPELINE_ON_GPU = True
                        except Exception as placement_error:
                            if not cfg.cpu_offload or not _is_cuda_memory_pressure(
                                placement_error
                            ):
                                raise
                            residency, key = _activate_cpu_offload(
                                pipeline,
                                cfg,
                                cache_enabled=cache_enabled,
                            )
                    else:
                        pipeline.enable_model_cpu_offload()
                        _IMAGE_PIPELINE_ON_GPU = False
                    if cache_enabled and _IMAGE_PIPELINE is None:
                        _IMAGE_PIPELINE = pipeline
                        _IMAGE_PIPELINE_KEY = key
                elif residency == "full_gpu" and not _IMAGE_PIPELINE_ON_GPU:
                    try:
                        pipeline.to("cuda")
                        _IMAGE_PIPELINE_ON_GPU = True
                    except Exception as placement_error:
                        if not cfg.cpu_offload or not _is_cuda_memory_pressure(
                            placement_error
                        ):
                            raise
                        residency, key = _activate_cpu_offload(
                            pipeline,
                            cfg,
                            cache_enabled=cache_enabled,
                        )

                def infer() -> Any:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    with torch.inference_mode():
                        return pipeline(
                            prompt=prompt,
                            width=width,
                            height=height,
                            generator=generator,
                            num_inference_steps=4,
                            guidance_scale=1.0,
                        )

                try:
                    result = infer()
                except Exception as inference_error:
                    if (
                        residency != "full_gpu"
                        or not cfg.cpu_offload
                        or not _is_cuda_memory_pressure(inference_error)
                    ):
                        raise
                    residency, key = _activate_cpu_offload(
                        pipeline,
                        cfg,
                        cache_enabled=cache_enabled,
                    )
                    result = infer()

                output = output_path.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(output)

                if not cache_enabled:
                    del pipeline
                    _IMAGE_PIPELINE_ON_GPU = False
                    _release_cuda()
                return output
        except ModelBackendError:
            raise
        except Exception as exc:
            with _IMAGE_LOCK:
                _clear_cached_pipeline()
            _release_cuda()
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc


ImageDiffusionAdapter.generate_image._mmm_adaptive_image_residency = True  # type: ignore[attr-defined]
