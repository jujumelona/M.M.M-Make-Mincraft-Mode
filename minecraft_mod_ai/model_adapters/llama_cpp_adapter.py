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


MODEL_LAYER_COUNTS = {
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": 40,
    "unsloth/Qwen3.6-27B-MTP-GGUF": 64,
    "unsloth/Qwen3.5-9B-MTP-GGUF": 32,
    "unsloth/gemma-4-26B-A4B-it-GGUF": 30,
    "unsloth/gemma-4-12b-it-GGUF": 48,
}


def _get_gguf_layer_count(model_path: str) -> int:
    """Read actual block_count / layer_count from GGUF binary header."""
    import struct
    try:
        with open(model_path, "rb") as f:
            header = f.read(4 * 1024 * 1024)
            pos = header.find(b".block_count")
            if pos != -1:
                for offset in range(pos + len(b".block_count"), min(len(header) - 8, pos + len(b".block_count") + 16)):
                    try:
                        val_type, val = struct.unpack("<II", header[offset:offset + 8])
                        if val_type in (4, 5, 8, 9, 10) and 4 <= val <= 128:
                            return val
                    except Exception:
                        pass
    except Exception:
        pass
    return 32


class LlamaCppAdapter(ModelAdapter):
    """Adapter: llama-cpp-python GGUF engine with GPU layer offloading."""

    _llm: Any = None
    _current_model_path: str | None = None

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._llm_instance: Any = None

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        require_package("llama-cpp-python")
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        try:
            # Resolve GGUF model path
            repo_id = cfg.model_id
            filename = cfg.extra.get("gguf_filename", "") or ""

            if os.path.exists(repo_id) and repo_id.endswith(".gguf"):
                model_path = repo_id
            else:
                hub_repo = repo_id
                if "/" not in hub_repo:
                    hub_repo = f"bartowski/{hub_repo}-GGUF"
                elif not hub_repo.endswith("-GGUF") and "GGUF" not in hub_repo:
                    repo_name = hub_repo.split('/')[-1]
                    hub_repo = f"bartowski/{repo_name}-GGUF"

                if not filename:
                    from huggingface_hub import list_repo_files
                    repo_files = list_repo_files(hub_repo)
                    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
                    if gguf_files:
                        q4_files = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f or "Q4_0" in f]
                        filename = q4_files[0] if q4_files else gguf_files[0]
                    else:
                        filename = f"{hub_repo.split('/')[-1]}.gguf"

                print(f"🚀 [llama.cpp] Fetching GGUF: {hub_repo} / {filename}...", flush=True)
                model_path = hf_hub_download(repo_id=hub_repo, filename=filename)

            if LlamaCppAdapter._llm is not None and LlamaCppAdapter._current_model_path != model_path:
                del LlamaCppAdapter._llm
                LlamaCppAdapter._llm = None
                LlamaCppAdapter._current_model_path = None
                gc.collect()

            if LlamaCppAdapter._llm is None:
                print(f"⚙️ [llama.cpp] Initializing GGUF model: {model_path}", flush=True)
                ctx_len = min(cfg.max_context, 16384)
                gpu_layers = cfg.extra.get("n_gpu_layers", -1)

                actual_total_layers = MODEL_LAYER_COUNTS.get(
                    cfg.model_id, _get_gguf_layer_count(model_path)
                )

                if gpu_layers == -1:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            total_free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                            # Dynamic KV cache calculation based on context length
                            dynamic_kv_mb = max(512.0, (ctx_len / 1024.0) * 128.0 + 512.0)
                            vram_for_weights = total_free_vram_mb - dynamic_kv_mb

                            if file_size_mb > vram_for_weights and vram_for_weights > 0:
                                weight_ratio = max(0.1, min(0.92, vram_for_weights / file_size_mb))
                                gpu_layers = max(1, int(actual_total_layers * weight_ratio))
                                print(
                                    f"💡 [llama.cpp] Dynamic GPU offload calculated: {gpu_layers}/{actual_total_layers} layers "
                                    f"(Free VRAM: {total_free_vram_mb:.0f}MB / Dynamic KV-Cache: {dynamic_kv_mb:.0f}MB / Model: {file_size_mb:.0f}MB)",
                                    flush=True,
                                )
                            else:
                                gpu_layers = -1
                                print(
                                    f"⚡ [llama.cpp] Full GPU load: All {actual_total_layers} layers on GPU "
                                    f"(Free VRAM: {total_free_vram_mb:.0f}MB / Model: {file_size_mb:.0f}MB)",
                                    flush=True,
                                )
                    except Exception:
                        pass

                # Retry loop to automatically handle edge-case VRAM pressure
                current_layers = gpu_layers
                while True:
                    try:
                        LlamaCppAdapter._llm = Llama(
                            model_path=model_path,
                            n_gpu_layers=current_layers,
                            n_ctx=ctx_len,
                            verbose=True,
                        )
                        break
                    except Exception as init_err:
                        err_msg = str(init_err).lower()
                        if current_layers != 0 and ("llama_context" in err_msg or "cuda" in err_msg or "out of memory" in err_msg):
                            base = actual_total_layers if current_layers == -1 else current_layers
                            new_layers = max(0, int(base * 0.85) - 1)
                            print(
                                f"⚠️ [llama.cpp] VRAM pressure detected with {current_layers} layers. "
                                f"Auto-reducing offload to {new_layers}/{actual_total_layers} layers and retrying...",
                                flush=True,
                            )
                            current_layers = new_layers
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise init_err

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
