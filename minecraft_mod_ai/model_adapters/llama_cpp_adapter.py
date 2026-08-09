"""llama.cpp open-source inference engine adapter.

Uses llama-cpp-python with GGUF quantization and GPU layer offloading
(n_gpu_layers). Ideal for running large models (14B/32B) on VRAM-constrained GPUs
by offloading layers between CPU RAM and GPU VRAM using C++ GGUF kernels.
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


class LlamaCppAdapter(ModelAdapter):
    """Adapter that delegates inference to the llama.cpp GGUF engine."""

    _llm: Any = None

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        try:
            require_package("llama_cpp")
            from llama_cpp import Llama

            if LlamaCppAdapter._llm is None:
                print(f"🦙 [llama.cpp] Loading GGUF model {cfg.model_id}...", flush=True)

                # Extract repo_id and filename if specified, or load directly
                model_path = cfg.model_id
                model_kwargs: dict[str, Any] = {
                    "n_gpu_layers": cfg.extra.get("n_gpu_layers", -1), # -1 = offload all layers to GPU
                    "n_ctx": min(cfg.max_context, 8192),
                    "verbose": False,
                }

                if "/" in cfg.model_id and not Path(cfg.model_id).exists():
                    filename = cfg.extra.get("filename", "*.gguf")
                    LlamaCppAdapter._llm = Llama.from_pretrained(
                        repo_id=cfg.model_id,
                        filename=filename,
                        **model_kwargs,
                    )
                else:
                    LlamaCppAdapter._llm = Llama(
                        model_path=model_path,
                        **model_kwargs,
                    )
                print(f"✅ [llama.cpp] Model loaded with GPU offloading.", flush=True)

            llm = LlamaCppAdapter._llm
            messages = [dict(m) for m in request.messages]

            print(f"🤖 [llama.cpp] Generating response...", flush=True)
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=cfg.max_new_tokens,
                temperature=0.0,
            )
            result = response["choices"][0]["message"]["content"].strip()
            print(f"✅ [llama.cpp] Generated {len(result)} chars.", flush=True)
            return result

        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc

    def close(self) -> None:
        pass
