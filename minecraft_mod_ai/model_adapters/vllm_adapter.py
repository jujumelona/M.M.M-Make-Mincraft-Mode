"""vLLM open-source inference engine adapter.

Uses vLLM's production-grade PagedAttention engine for high-throughput,
low-latency LLM inference. Supports cpu_offload_gb for CPU RAM offloading
and cleans up GPU VRAM on close.
"""
from __future__ import annotations

import gc
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
    """Adapter that delegates all LLM inference to the vLLM engine."""

    _llm: Any = None
    _current_model_id: str | None = None

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            require_package("vllm")
            from vllm import LLM, SamplingParams
            import torch

            # Unload previous model from VRAM if model changed
            if VLLMAdapter._llm is not None and VLLMAdapter._current_model_id != cfg.model_id:
                print(f"🧹 [vLLM] Unloading previous model ({VLLMAdapter._current_model_id}) from VRAM...", flush=True)
                del VLLMAdapter._llm
                VLLMAdapter._llm = None
                VLLMAdapter._current_model_id = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # Initialize vLLM engine if not loaded
            if VLLMAdapter._llm is None:
                print(f"🚀 [vLLM Engine] Initializing and loading {cfg.model_id}...", flush=True)

                vllm_kwargs: dict[str, Any] = {
                    "model": cfg.model_id,
                    "trust_remote_code": True,
                    "max_model_len": min(cfg.max_context, 8192),
                    "gpu_memory_utilization": 0.85,
                    "dtype": "half" if cfg.torch_dtype == "float16" else "auto",
                }

                # Optional CPU offloading support in vLLM
                if cfg.cpu_offload:
                    vllm_kwargs["cpu_offload_gb"] = 4

                if cfg.quantization == "bnb_4bit":
                    vllm_kwargs["quantization"] = "bitsandbytes"
                    vllm_kwargs["load_format"] = "bitsandbytes"
                elif cfg.quantization:
                    vllm_kwargs["quantization"] = cfg.quantization

                VLLMAdapter._llm = LLM(**vllm_kwargs)
                VLLMAdapter._current_model_id = cfg.model_id
                print(f"✅ [vLLM Engine] {cfg.model_id} loaded successfully into vLLM engine.", flush=True)

            llm = VLLMAdapter._llm
            tokenizer = llm.get_tokenizer()

            # Format chat prompt
            messages = [dict(m) for m in request.messages]
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

            print(f"🤖 [vLLM Engine] Generating tokens (max_new_tokens={cfg.max_new_tokens})...", flush=True)
            outputs = llm.generate([prompt], sampling_params)
            result = outputs[0].outputs[0].text.strip()
            print(f"✅ [vLLM Engine] Finished generation ({len(result)} chars).", flush=True)
            return result

        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc

    def close(self) -> None:
        if VLLMAdapter._llm is not None:
            del VLLMAdapter._llm
            VLLMAdapter._llm = None
            VLLMAdapter._current_model_id = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
