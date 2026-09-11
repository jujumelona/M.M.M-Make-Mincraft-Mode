from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import ModelBackendError, ModelConfigurationError, _release_cuda, preflight_cuda, require_package, torch_dtype

_IMAGE_LOCK = threading.RLock()
_IMAGE_PIPELINE: Any | None = None
_IMAGE_PIPELINE_KEY: tuple[Any, ...] | None = None
_IMAGE_PIPELINE_ON_GPU = False


@dataclass(frozen=True)
class ImageGenerationConfig:
    model_id: str
    quantization: str | None
    torch_dtype: str
    cpu_offload: bool
    lora_model_id: str
    lora_weight_name: str
    lora_adapter_name: str
    lora_scale: float
    lora_trigger: str
    num_inference_steps: int
    guidance_scale: float
    candidate_count: int

    @classmethod
    def from_adapter_config(cls, config: Any) -> "ImageGenerationConfig":
        extra: Mapping[str, Any] = config.extra if isinstance(config.extra, Mapping) else {}
        result = cls(
            model_id=str(config.model_id or "").strip(),
            quantization=str(config.quantization).strip() if config.quantization else None,
            torch_dtype=str(config.torch_dtype or "auto"),
            cpu_offload=bool(config.cpu_offload),
            lora_model_id=str(extra.get("lora_model_id") or "").strip(),
            lora_weight_name=str(extra.get("lora_weight_name") or "").strip(),
            lora_adapter_name=str(extra.get("lora_adapter_name") or "mmm_image_lora").strip(),
            lora_scale=float(extra.get("lora_scale", 1.0)),
            lora_trigger=str(extra.get("lora_trigger") or "").strip(),
            num_inference_steps=int(extra.get("num_inference_steps", 4)),
            guidance_scale=float(extra.get("guidance_scale", 1.0)),
            candidate_count=int(extra.get("candidate_count", 4)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.model_id:
            raise ModelConfigurationError("Image model_id is required.")
        if self.quantization not in {None, "bnb_4bit", "bnb_4bit_nf4"}:
            raise ModelConfigurationError(f"Unsupported image quantization: {self.quantization!r}.")
        if self.num_inference_steps < 1 or self.candidate_count < 1 or self.guidance_scale < 0:
            raise ModelConfigurationError("Image inference settings are invalid.")
        if bool(self.lora_model_id) != bool(self.lora_weight_name):
            raise ModelConfigurationError("LoRA model_id and weight_name must be configured together.")
        if self.lora_model_id and not self.lora_trigger:
            raise ModelConfigurationError("Configured image LoRA requires lora_trigger.")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() not in {"0", "false", "no", "off"}


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


def _accelerate_memory_budget(torch_module: Any) -> dict[Any, str]:
    total = int(torch_module.cuda.get_device_properties(0).total_memory)
    reserve = max(2 * 1024**3, int(total * 0.14))
    gpu = max(4 * 1024**3, total - reserve)
    try:
        import psutil
        cpu_available = int(psutil.virtual_memory().available)
    except Exception:
        cpu_available = 8 * 1024**3
    cpu = max(2 * 1024**3, int(cpu_available * 0.8))
    return {0: f"{gpu // 1024**2}MiB", "cpu": f"{cpu // 1024**2}MiB"}


def _load_pipeline(config: Any, p: ImageGenerationConfig) -> Any:
    if p.quantization in {"bnb_4bit", "bnb_4bit_nf4"}:
        require_package("diffusers", minimum="0.39.0")
        require_package("transformers", minimum="4.56.0")
        require_package("accelerate", minimum="1.0.0")
        require_package("bitsandbytes", minimum="0.45.0")
        import torch
        from diffusers import BitsAndBytesConfig as DBits
        from diffusers import Flux2KleinPipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        from transformers import BitsAndBytesConfig as TBits

        dtype = torch_dtype(p.torch_dtype)
        if dtype == "auto":
            dtype = torch.float16
        quant = PipelineQuantizationConfig(quant_mapping={
            "transformer": DBits(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
            "text_encoder": TBits(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
        })
        pipeline = Flux2KleinPipeline.from_pretrained(
            p.model_id, torch_dtype=dtype, quantization_config=quant, device_map="auto",
            max_memory=_accelerate_memory_budget(torch), low_cpu_mem_usage=True, trust_remote_code=False,
        )
    else:
        require_package("diffusers", minimum="0.39.0")
        require_package("transformers", minimum="4.56.0")
        require_package("accelerate", minimum="1.0.0")
        from diffusers import DiffusionPipeline
        pipeline = DiffusionPipeline.from_pretrained(p.model_id, torch_dtype=torch_dtype(p.torch_dtype), trust_remote_code=False)
        if config.cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")

    if p.lora_model_id:
        require_package("peft", minimum="0.17.0")
        loader = getattr(pipeline, "load_lora_weights", None)
        if not callable(loader):
            raise ModelConfigurationError("Installed image pipeline does not support LoRA loading.")
        loader(p.lora_model_id, weight_name=p.lora_weight_name, adapter_name=p.lora_adapter_name)
        setter = getattr(pipeline, "set_adapters", None)
        if callable(setter):
            setter(p.lora_adapter_name, adapter_weights=p.lora_scale)
    progress = getattr(pipeline, "set_progress_bar_config", None)
    if callable(progress):
        progress(disable=True)
    return pipeline


def _profile_key(p: ImageGenerationConfig) -> tuple[Any, ...]:
    return (p.model_id, p.quantization, p.torch_dtype, p.lora_model_id, p.lora_weight_name,
            p.lora_adapter_name, p.lora_scale, p.num_inference_steps, p.guidance_scale)


def _clear_cached_pipeline() -> None:
    global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY, _IMAGE_PIPELINE_ON_GPU
    _IMAGE_PIPELINE = None
    _IMAGE_PIPELINE_KEY = None
    _IMAGE_PIPELINE_ON_GPU = False


def finish_image_shard() -> None:
    global _IMAGE_PIPELINE_ON_GPU
    with _IMAGE_LOCK:
        pipeline = _IMAGE_PIPELINE
        if not _env_bool("MMM_IMAGE_CACHE_ACROSS_SHARDS", False):
            _clear_cached_pipeline()
        elif pipeline is not None and _IMAGE_PIPELINE_ON_GPU:
            pipeline.to("cpu")
            _IMAGE_PIPELINE_ON_GPU = False
    _release_cuda()


class ImageDiffusionAdapter:
    """Typed image backend. Registry owns model, quantization, LoRA and inference settings."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.profile = ImageGenerationConfig.from_adapter_config(config)

    def generate_image(self, *, prompt: str, output_path: Path, width: int = 512, height: int = 512, seed: int = 0) -> Path:
        global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY, _IMAGE_PIPELINE_ON_GPU
        try:
            if not str(prompt).strip():
                raise ModelConfigurationError("Image prompt is empty.")
            if width % 16 or height % 16 or not (256 <= width <= 1024 and 256 <= height <= 1024):
                raise ModelConfigurationError("Image dimensions must be 256-1024 and divisible by 16.")
            preflight_cuda(self.config)
            import torch
            key = _profile_key(self.profile)
            cache_enabled = _env_bool("MMM_IMAGE_PIPELINE_CACHE", True)
            with _IMAGE_LOCK:
                pipeline = _IMAGE_PIPELINE if cache_enabled and _IMAGE_PIPELINE_KEY == key else None
                if pipeline is None:
                    _clear_cached_pipeline()
                    _release_cuda()
                    pipeline = _load_pipeline(self.config, self.profile)
                    _IMAGE_PIPELINE_ON_GPU = not bool(self.config.cpu_offload)
                    if cache_enabled:
                        _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY = pipeline, key
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
                with torch.inference_mode():
                    result = pipeline(prompt=str(prompt), width=int(width), height=int(height), generator=generator,
                                      num_inference_steps=self.profile.num_inference_steps,
                                      guidance_scale=self.profile.guidance_scale)
                images = getattr(result, "images", None)
                if not images:
                    raise ModelConfigurationError("Image pipeline returned no image.")
                output = Path(output_path).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                images[0].convert("RGBA").save(output, format="PNG", optimize=False)
                if not cache_enabled:
                    _clear_cached_pipeline()
                    _release_cuda()
                return output
        except ModelBackendError:
            raise
        except Exception as exc:
            with _IMAGE_LOCK:
                _clear_cached_pipeline()
            _release_cuda()
            raise ModelBackendError(role=self.config.role, model_id=self.config.model_id, cause=exc) from exc


ImageDiffusionAdapter.generate_image._mmm_adaptive_image_residency = True  # type: ignore[attr-defined]
__all__ = ["ImageDiffusionAdapter", "ImageGenerationConfig", "finish_image_shard"]
