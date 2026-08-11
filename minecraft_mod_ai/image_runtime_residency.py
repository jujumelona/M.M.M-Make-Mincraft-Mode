from __future__ import annotations

import os
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any


def _full_gpu_threshold_mb(config: Any) -> int:
    raw = os.environ.get("MMM_IMAGE_FULL_GPU_MIN_FREE_MB", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    # FLUX.2 Klein 4B is documented around the 13 GB VRAM class. Keep additional
    # headroom above the configured preflight floor rather than attempting a full
    # residency move at the exact minimum.
    return max(14_000, int(config.min_free_vram_mb) + 1_000)


def install() -> None:
    from . import complete_orchestrator_services as services
    from . import model_router as router_module
    from . import model_runtime_performance as runtime
    from .model_adapters import base as base_module
    from .model_adapters.image_diffusion import ImageDiffusionAdapter
    from .model_adapters.llama_cpp_adapter import LlamaCppAdapter

    _install_adaptive_image_generation(
        ImageDiffusionAdapter,
        LlamaCppAdapter,
        runtime,
        base_module,
    )
    _install_pipeline_parking(
        services,
        router_module,
        runtime,
        base_module,
    )


def _install_adaptive_image_generation(
    cls: Any,
    llama_cls: Any,
    runtime: Any,
    base_module: Any,
) -> None:
    original = cls.generate_image
    if getattr(original, "_mmm_adaptive_image_residency", False):
        return

    @wraps(original)
    def adaptive_generate_image(
        self: Any,
        *,
        prompt: str,
        output_path: Path,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Path:
        cfg = self.config
        try:
            base_module.require_package("diffusers", minimum="0.20.0")
            base_module.require_package("transformers", minimum="4.52.0")
            base_module.require_package("accelerate", minimum="1.0.0")
            if not prompt.strip():
                raise base_module.ModelConfigurationError("Image prompt is empty.")
            if (
                width % 16
                or height % 16
                or not (256 <= width <= 1024 and 256 <= height <= 1024)
            ):
                raise base_module.ModelConfigurationError(
                    "Image dimensions must be 256-1024 and divisible by 16."
                )

            # Router owns the exclusive GPU lock before entering this method.
            runtime._evict_llama(llama_cls, base_module._release_cuda)
            base_module.preflight_cuda(cfg)

            import torch
            from diffusers import DiffusionPipeline

            free_bytes, _ = torch.cuda.mem_get_info()
            free_mb = int(free_bytes / (1024 * 1024))
            use_full_gpu = (
                not cfg.cpu_offload
                or free_mb >= _full_gpu_threshold_mb(cfg)
            )
            residency = "full_gpu" if use_full_gpu else "cpu_offload"
            cache_enabled = runtime._env_bool(
                "MMM_IMAGE_PIPELINE_CACHE",
                True,
            )
            key = (cfg.model_id, cfg.torch_dtype, residency)

            with runtime._IMAGE_LOCK:
                pipeline = (
                    runtime._IMAGE_PIPELINE
                    if cache_enabled and runtime._IMAGE_PIPELINE_KEY == key
                    else None
                )
                if pipeline is None:
                    if runtime._IMAGE_PIPELINE is not None:
                        runtime._IMAGE_PIPELINE = None
                        runtime._IMAGE_PIPELINE_KEY = None
                        setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False)
                        base_module._release_cuda()
                    pipeline = DiffusionPipeline.from_pretrained(
                        cfg.model_id,
                        torch_dtype=base_module.torch_dtype(cfg.torch_dtype),
                        trust_remote_code=False,
                    )
                    if residency == "full_gpu":
                        pipeline.to("cuda")
                        setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", True)
                    else:
                        pipeline.enable_model_cpu_offload()
                        setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False)
                    progress = getattr(pipeline, "set_progress_bar_config", None)
                    if callable(progress):
                        progress(disable=True)
                    if cache_enabled:
                        runtime._IMAGE_PIPELINE = pipeline
                        runtime._IMAGE_PIPELINE_KEY = key
                elif (
                    residency == "full_gpu"
                    and not bool(getattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False))
                ):
                    # The previous asset shard parked the cached weights on CPU.
                    # Moving an already-instantiated pipeline is far cheaper than
                    # rebuilding it from safetensors and reconstructing modules.
                    pipeline.to("cuda")
                    setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", True)

                generator = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    result = pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        generator=generator,
                        num_inference_steps=4,
                        guidance_scale=1.0,
                    )
                output = output_path.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(output)

                if not cache_enabled:
                    del pipeline
                    setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False)
                    base_module._release_cuda()
                return output
        except base_module.ModelBackendError:
            raise
        except Exception as exc:
            with runtime._IMAGE_LOCK:
                runtime._IMAGE_PIPELINE = None
                runtime._IMAGE_PIPELINE_KEY = None
                setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False)
            base_module._release_cuda()
            raise base_module.ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc

    adaptive_generate_image._mmm_adaptive_image_residency = True
    cls.generate_image = adaptive_generate_image


def _park_cached_image_pipeline(runtime: Any, base_module: Any) -> None:
    with runtime._IMAGE_LOCK:
        pipeline = runtime._IMAGE_PIPELINE
        key = runtime._IMAGE_PIPELINE_KEY
        if (
            pipeline is not None
            and key is not None
            and key[-1] == "full_gpu"
            and bool(getattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False))
        ):
            pipeline.to("cpu")
            setattr(runtime, "_IMAGE_PIPELINE_ON_GPU", False)
    base_module._release_cuda()


def _install_pipeline_parking(
    services: Any,
    router_module: Any,
    runtime: Any,
    base_module: Any,
) -> None:
    original = services.generate_assets
    if getattr(original, "_mmm_adaptive_image_gpu_session", False):
        return

    @wraps(original)
    def adaptive_asset_session(router: Any, *args: Any, **kwargs: Any):
        config = router.registry.role(router.profile, "image_generator")
        local_exclusive = (
            config.provider == "local"
            and config.adapter == "image_diffusion"
            and config.exclusive_gpu
        )
        if not local_exclusive:
            return original(router, *args, **kwargs)

        # Keep the outer lock across the already-installed asset-session wrapper so
        # cached full-GPU weights are parked before a local LLM can acquire the GPU.
        with router_module._GPU_EXCLUSIVE_LOCK:
            try:
                return original(router, *args, **kwargs)
            finally:
                _park_cached_image_pipeline(runtime, base_module)

    adaptive_asset_session._mmm_adaptive_image_gpu_session = True
    services.generate_assets = adaptive_asset_session
