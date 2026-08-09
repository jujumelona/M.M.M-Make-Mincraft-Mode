from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .base import (
    AdapterConfig,
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
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._session_active = False

    @contextmanager
    def generation_session(self):
        """Pin one lazy-loaded processor/model pair for paginated generation."""

        if self._session_active:
            raise ModelConfigurationError(
                "Transformers multimodal generation session is already active."
            )
        self._session_active = True
        try:
            yield self
        finally:
            self._session_active = False
            self.close()

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            require_package("transformers", minimum="4.57.0")
            require_package("accelerate", minimum="1.0.0")
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    cfg.model_id,
                    trust_remote_code=False,
                )
            processor = self._processor
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
            input_tokens = int(inputs["input_ids"].shape[-1])
            requested_tokens = input_tokens + cfg.max_new_tokens
            if requested_tokens > cfg.max_context:
                raise ModelConfigurationError(
                    "Rendered multimodal request exceeds the model context: "
                    f"{input_tokens} input + {cfg.max_new_tokens} reserved output "
                    f"> max_context={cfg.max_context}."
                )

            # Validate the fully rendered request before loading multi-gigabyte
            # model weights.  An oversized prompt is a configuration error, not
            # a reason to spend GPU memory and tens of seconds loading a model.
            if self._model is None:
                preflight_cuda(cfg)
                kwargs: dict[str, Any] = {
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch_dtype(cfg.torch_dtype),
                    "trust_remote_code": False,
                }
                qconfig = quantization_config(cfg)
                if qconfig is not None:
                    kwargs["quantization_config"] = qconfig
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    cfg.model_id,
                    **kwargs,
                )
            model = self._model
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                )
            generated = output[:, input_tokens:]
            return processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc
        finally:
            if not self._session_active:
                self.close()

    def close(self) -> None:
        model = self._model
        processor = self._processor
        self._model = None
        self._processor = None
        if model is not None:
            del model
        if processor is not None:
            del processor
        super().close()
