from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Any, Iterator

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


class TransformersTextAdapter(ModelAdapter):
    """Transformers text backend with explicit generation-session residency."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._backend_lock = threading.RLock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._session_depth = 0

    @contextmanager
    def generation_session(self) -> Iterator[TransformersTextAdapter]:
        """Keep tokenizer/model resident across bounded calls in one router stage."""

        with self._backend_lock:
            self._session_depth += 1
        try:
            yield self
        finally:
            with self._backend_lock:
                self._session_depth -= 1
                should_close = self._session_depth == 0
            if should_close:
                self.close()

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        require_package("transformers", minimum="4.52.0")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=False,
        )
        return self._tokenizer

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        cfg = self.config
        require_package("transformers", minimum="4.52.0")
        require_package("accelerate", minimum="1.0.0")
        preflight_cuda(cfg)
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "dtype": torch_dtype(cfg.torch_dtype),
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
        }
        if cfg.cpu_offload:
            kwargs["offload_folder"] = "offload_dir"
        qconfig = quantization_config(cfg)
        if qconfig is not None:
            kwargs["quantization_config"] = qconfig
        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        return self._model

    def _validate_context(self, tokenizer: Any, request: GenerationRequest) -> dict[str, Any]:
        cfg = self.config
        rendered = tokenizer.apply_chat_template(
            list(request.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=False,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        if cfg.max_context > 0 and input_tokens + cfg.max_new_tokens > cfg.max_context:
            raise ModelConfigurationError(
                "Rendered text request exceeds the model context: "
                f"{input_tokens} input + {cfg.max_new_tokens} reserved output "
                f"> max_context={cfg.max_context}. "
                "Split the request into bounded, verifiable pages instead of "
                "discarding prompt tokens."
            )
        if cfg.max_input_tokens > 0 and input_tokens > cfg.max_input_tokens:
            raise ModelConfigurationError(
                "Rendered text page exceeds this hardware profile's per-call "
                f"input budget: {input_tokens} input tokens > "
                f"max_input_tokens={cfg.max_input_tokens}. Split this stage "
                "into additional pages."
            )
        return inputs

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            import torch

            with self._backend_lock:
                tokenizer = self._ensure_tokenizer()
                inputs = self._validate_context(tokenizer, request)
                model = self._ensure_model()
                device = next(model.parameters()).device
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                generation_inputs = dict(inputs)
                with torch.inference_mode():
                    while True:
                        try:
                            output = model.generate(
                                **generation_inputs,
                                max_new_tokens=cfg.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                            break
                        except ValueError as exc:
                            message = str(exc)
                            if "are not used by the model:" not in message:
                                raise
                            unused_keys = re.findall(r"'([^']+)'", message)
                            removed = False
                            for key in unused_keys:
                                if key in generation_inputs:
                                    generation_inputs.pop(key)
                                    removed = True
                            if not removed:
                                raise
                generated = output[0, inputs["input_ids"].shape[1] :]
                return tokenizer.decode(generated, skip_special_tokens=True).strip()
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc
        finally:
            with self._backend_lock:
                should_close = self._session_depth == 0
            if should_close:
                self.close()

    def close(self) -> None:
        with self._backend_lock:
            self._model = None
            self._tokenizer = None
        super().close()
