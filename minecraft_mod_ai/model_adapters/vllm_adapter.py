"""vLLM open-source inference engine adapter.

Uses vLLM's production-grade PagedAttention engine for high-throughput,
low-latency LLM inference. No custom model loading or generation code —
everything is delegated to the vLLM library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    AdapterConfig,
    GenerationRequest,
    ModelAdapter,
    ModelBackendError,
    ModelConfigurationError,
    require_package,
)


class VLLMAdapter(ModelAdapter):
    """Adapter that delegates all inference to the vLLM engine."""

    _llm: Any = None
    _tokenizer: Any = None

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            require_package("vllm")
            from vllm import LLM, SamplingParams

            if VLLMAdapter._llm is None or VLLMAdapter._llm.llm_engine.model_config.model != cfg.model_id:
                # Release previous engine if model changed
                if VLLMAdapter._llm is not None:
                    del VLLMAdapter._llm
                    VLLMAdapter._llm = None
                    import gc, torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"🚀 [vLLM] Loading {cfg.model_id}...", flush=True)

                vllm_kwargs: dict[str, Any] = {
                    "model": cfg.model_id,
                    "trust_remote_code": True,
                    "max_model_len": min(cfg.max_context, 8192),
                    "gpu_memory_utilization": 0.90,
                    "dtype": "half",
                }
                if cfg.quantization == "bnb_4bit":
                    vllm_kwargs["quantization"] = "bitsandbytes"
                    vllm_kwargs["load_format"] = "bitsandbytes"

                VLLMAdapter._llm = LLM(**vllm_kwargs)
                print(f"✅ [vLLM] {cfg.model_id} loaded.", flush=True)

            llm = VLLMAdapter._llm
            tokenizer = llm.get_tokenizer()

            # Build chat messages
            messages = [dict(m) for m in request.messages]

            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            sampling_params = SamplingParams(
                max_tokens=cfg.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )

            print(f"🤖 [vLLM] Generating (max_tokens={cfg.max_new_tokens})...", flush=True)
            outputs = llm.generate([prompt], sampling_params)
            result = outputs[0].outputs[0].text.strip()
            print(f"✅ [vLLM] Generated {len(result)} chars.", flush=True)
            return result

        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc

    def close(self) -> None:
        pass
