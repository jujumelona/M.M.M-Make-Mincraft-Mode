from __future__ import annotations

from pathlib import Path

from .base import ModelBackendError, ModelConfigurationError, preflight_cuda, require_package, torch_dtype


class ImageDiffusionAdapter:
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
        cfg = self.config
        try:
            require_package("diffusers", minimum="0.20.0")
            require_package("transformers", minimum="4.52.0")
            require_package("accelerate", minimum="1.0.0")
            preflight_cuda(cfg)
            import torch
            from diffusers import DiffusionPipeline

            if not prompt.strip():
                raise ModelConfigurationError("Image prompt is empty.")
            if width % 16 or height % 16 or not (256 <= width <= 1024 and 256 <= height <= 1024):
                raise ModelConfigurationError("Image dimensions must be 256-1024 and divisible by 16.")
            pipeline = DiffusionPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=torch_dtype(cfg.torch_dtype),
                trust_remote_code=False,
            )
            try:
                if cfg.cpu_offload:
                    pipeline.enable_model_cpu_offload()
                else:
                    pipeline.to("cuda")
                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    generator=generator,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                )
                output_path = output_path.expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(output_path)
                return output_path
            finally:
                del pipeline
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc
        finally:
            from .base import _release_cuda

            _release_cuda()
