"""llama.cpp GGUF inference engine adapter for high-performance 32B model execution.

Supports GGUF models with automatic GPU layer offloading (n_gpu_layers=-1).
Downloads GGUF weights via huggingface_hub on demand.
"""
from __future__ import annotations

import gc
import os
from typing import Any

from .base import (
    AdapterConfig,
    GenerationRequest,
    ModelAdapter,
    ModelBackendError,
    require_package,
)


class LlamaCppAdapter(ModelAdapter):
    """Adapter: llama-cpp-python GGUF engine with GPU layer offloading."""

    _llm: Any = None
    _current_model_path: str | None = None

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._llm_instance: Any = None

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        require_package("llama_cpp")
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        try:
            # Resolve GGUF model path
            repo_id = cfg.model_id
            filename = getattr(cfg, "gguf_filename", "") or ""

            if os.path.exists(repo_id) and repo_id.endswith(".gguf"):
                model_path = repo_id
            else:
                hub_repo = repo_id
                if "/" not in hub_repo:
                    hub_repo = f"bartowski/{hub_repo}-GGUF"

                if not filename:
                    from huggingface_hub import list_repo_files
                    repo_files = list_repo_files(hub_repo)
                    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
                    if gguf_files:
                        # Prefer Q4_K_M or first gguf file
                        q4_files = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f]
                        filename = q4_files[0] if q4_files else gguf_files[0]
                    else:
                        filename = f"{repo_id.split('/')[-1]}.gguf"

                print(f"🚀 [llama.cpp] Fetching GGUF: {hub_repo} / {filename}...", flush=True)
                model_path = hf_hub_download(repo_id=hub_repo, filename=filename)

            if LlamaCppAdapter._llm is not None and LlamaCppAdapter._current_model_path != model_path:
                del LlamaCppAdapter._llm
                LlamaCppAdapter._llm = None
                LlamaCppAdapter._current_model_path = None
                gc.collect()

            if LlamaCppAdapter._llm is None:
                print(f"⚙️ [llama.cpp] Initializing GGUF model with GPU offloading (n_gpu_layers=-1)...", flush=True)
                ctx_len = min(cfg.max_context, 16384)
                LlamaCppAdapter._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,  # Offload max possible layers to T4 GPU VRAM
                    n_ctx=ctx_len,
                    verbose=False,
                )
                LlamaCppAdapter._current_model_path = model_path
                print(f"✅ [llama.cpp] Model successfully loaded.", flush=True)

            llm = LlamaCppAdapter._llm
            messages = [dict(m) for m in request.messages]

            # Format messages into chat prompt
            try:
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=cfg.max_new_tokens,
                    temperature=0.0,
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception:
                # Fallback to direct raw prompt format
                prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                response = llm(
                    prompt_text,
                    max_tokens=cfg.max_new_tokens,
                    temperature=0.0,
                    stop=["<|endoftext|>", "<|im_end|>"],
                )
                return response["choices"][0]["text"].strip()

        except Exception as exc:
            raise ModelBackendError(role=cfg.role, model_id=cfg.model_id, cause=exc) from exc

    def close(self) -> None:
        if LlamaCppAdapter._llm is not None:
            del LlamaCppAdapter._llm
            LlamaCppAdapter._llm = None
            LlamaCppAdapter._current_model_path = None
            gc.collect()
