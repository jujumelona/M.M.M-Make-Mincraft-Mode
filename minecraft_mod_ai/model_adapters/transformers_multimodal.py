from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import (
    GenerationRequest,
    ModelAdapter,
    ModelBackendError,
    ModelConfigurationError,
    preflight_cuda,
    quantization_config,
    require_package,
    torch_dtype,
)


def _normalize_messages(
    messages: list[Mapping[str, Any]], media_paths: tuple[Path, ...]
) -> list[dict[str, Any]]:
    normalized = [dict(message) for message in messages]
    if not media_paths:
        return normalized
    if not normalized or normalized[-1].get("role") != "user":
        raise ModelConfigurationError("Media can only be attached to a final user message.")
    user_text = normalized[-1].get("content", "")
    if not isinstance(user_text, str):
        raise ModelConfigurationError("The final user message must contain text.")
    content: list[dict[str, str]] = []
    for path in media_paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ModelConfigurationError(f"Media file does not exist: {resolved}")
        content.append({"type": "image", "url": resolved.as_uri()})
    content.append({"type": "text", "text": user_text})
    normalized[-1] = {"role": "user", "content": content}
    return normalized


class TransformersMultimodalAdapter(ModelAdapter):
    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            require_package("transformers", minimum="4.57.0")
            require_package("accelerate", minimum="1.0.0")
            preflight_cuda(cfg)
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=False)
            kwargs: dict[str, Any] = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "torch_dtype": torch_dtype(cfg.torch_dtype),
                "trust_remote_code": False,
            }
            qconfig = quantization_config(cfg)
            if qconfig is not None:
                kwargs["quantization_config"] = qconfig
            model = AutoModelForMultimodalLM.from_pretrained(cfg.model_id, **kwargs)
            try:
                messages = _normalize_messages(list(request.messages), request.media_paths)
                template_kwargs: dict[str, Any] = {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                }
                # Qwen3.5 emits a separate <think> response by default.  That is
                # useful for prose, but a structured planner response must begin
                # with its contract JSON so a reasoning draft cannot be mistaken
                # for the actual plan.  The flag is a documented Qwen3.5 chat
                # template option, so do not pass it to unrelated models.
                if (
                    request.response_format == "json"
                    and cfg.model_id.lower().startswith("qwen/qwen3.5")
                ):
                    template_kwargs["enable_thinking"] = False
                inputs = processor.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
                if inputs["input_ids"].shape[-1] > cfg.max_context:
                    raise ModelConfigurationError(
                        f"Rendered multimodal prompt exceeds max_context={cfg.max_context}."
                    )
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=False,
                    )
                generated = output[:, inputs["input_ids"].shape[-1] :]
                return processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0].strip()
            finally:
                del model
                del processor
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc
        finally:
            self.close()
