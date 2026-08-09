"""vLLM open-source inference engine adapter with automatic PyTorch fallback.

Uses vLLM's production-grade PagedAttention engine for high-throughput LLM inference.
If vLLM fails due to CUDA library mismatch (e.g. libcudart.so.13), it seamlessly
falls back to standard PyTorch/HuggingFace inference to prevent runtime failure.
"""
from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any

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


class VLLMAdapter(ModelAdapter):
    """Adapter that delegates LLM inference to vLLM, with PyTorch fallback."""

    _llm: Any = None
    _current_model_id: str | None = None

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._fallback_processor: Any = None
        self._fallback_model: Any = None

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

        except Exception as exc:
            msg = str(exc)
            if "libcudart" in msg or "vllm" in msg or "CUDA" in msg or isinstance(exc, (ImportError, RuntimeError, OSError)):
                print(f"ℹ️ [vLLM Notice] vLLM engine unavailable ({exc}). Using standard PyTorch pipeline for {cfg.model_id}...", flush=True)
                return self._fallback_generate(request)
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc

    def _fallback_generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, AutoTokenizer

            preflight_cuda(cfg)

            if self._fallback_processor is None:
                try:
                    self._fallback_processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
                except Exception:
                    self._fallback_processor = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)

            processor = self._fallback_processor
            messages = [dict(m) for m in request.messages]
            tokenizer_or_proc = getattr(processor, "tokenizer", processor)

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                rendered = tokenizer_or_proc.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = tokenizer_or_proc(rendered, return_tensors="pt")

            input_tokens = int(inputs["input_ids"].shape[-1])

            if self._fallback_model is None:
                kwargs: dict[str, Any] = {
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "dtype": torch_dtype(cfg.torch_dtype),
                    "trust_remote_code": True,
                }
                qconfig = quantization_config(cfg)
                if qconfig is not None:
                    kwargs["quantization_config"] = qconfig

                model_loaded = None
                for name in ("AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModel"):
                    auto_cls = getattr(transformers, name, None)
                    if auto_cls is not None:
                        try:
                            model_loaded = auto_cls.from_pretrained(cfg.model_id, **kwargs)
                            break
                        except Exception:
                            continue
                if model_loaded is None:
                    model_loaded = transformers.AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
                self._fallback_model = model_loaded

            model = self._fallback_model
            device = next(model.parameters()).device
            inputs = inputs.to(device)

            gen_inputs = dict(inputs)
            with torch.inference_mode():
                while True:
                    try:
                        output = model.generate(
                            **gen_inputs,
                            max_new_tokens=cfg.max_new_tokens,
                            do_sample=False,
                        )
                        break
                    except ValueError as ve:
                        ve_msg = str(ve)
                        if "are not used by the model:" in ve_msg:
                            unused_keys = re.findall(r"'([^']+)'", ve_msg)
                            popped = False
                            for key in unused_keys:
                                if key in gen_inputs:
                                    gen_inputs.pop(key, None)
                                    popped = True
                            if popped:
                                continue
                        raise
            generated = output[:, input_tokens:]
            return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
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
        if self._fallback_model is not None:
            del self._fallback_model
            self._fallback_model = None
        if self._fallback_processor is not None:
            del self._fallback_processor
            self._fallback_processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
