from __future__ import annotations

from typing import Any

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
    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            require_package(
                "transformers",
                minimum="4.48.0",
                maximum_exclusive="5.0.0",
            )
            require_package("accelerate", minimum="1.0.0")
            preflight_cuda(cfg)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=False)
            rendered = tokenizer.apply_chat_template(
                list(request.messages), tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=False,
            )
            input_tokens = int(inputs["input_ids"].shape[-1])
            if input_tokens + cfg.max_new_tokens > cfg.max_context:
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
            kwargs: dict[str, Any] = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "dtype": torch_dtype(cfg.torch_dtype),
                "attn_implementation": "sdpa",
                "trust_remote_code": False,
            }
            qconfig = quantization_config(cfg)
            if qconfig is not None:
                kwargs["quantization_config"] = qconfig
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
            try:
                device = next(model.parameters()).device
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output[0, inputs["input_ids"].shape[1] :]
                return tokenizer.decode(generated, skip_special_tokens=True).strip()
            finally:
                del model
                del tokenizer
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc
        finally:
            self.close()
