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
    "google/gemma-4-26B-A4B-it-qat-q4_0-gguf": 30,
    "google/gemma-4-12B-it-qat-q4_0-gguf": 48,
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
        
        # 1. Check if llama-server (C++ binary with MTP / Draft Speculative Decoding) is active on local HTTP port
        server_url = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8910/v1").rstrip("/")
        try:
            import requests
            check_resp = requests.get(f"{server_url}/models", timeout=0.3)
            if check_resp.status_code == 200:
                print(f"🚀 [llama-server MTP] C++ MTP 백엔드 연동 중 ({server_url})...", flush=True)
                messages = [dict(m) for m in request.messages]
                payload = {
                    "model": "local",
                    "messages": messages,
                    "max_tokens": cfg.max_new_tokens,
                    "temperature": 0.0,
                }
                if getattr(request, "response_format", None) == "json":
                    payload["response_format"] = {"type": "json_object"}
                
                resp = requests.post(f"{server_url}/chat/completions", json=payload, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass  # Fallback to in-memory python llama-cpp-python adapter below

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
                elif not hub_repo.endswith("-GGUF") and "GGUF" not in hub_repo and "gguf" not in hub_repo:
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
                # Keep FULL max_context (16k) without reducing context size!
                ctx_len = cfg.max_context

                try:
                    import torch
                except ImportError:
                    torch = None

                actual_total_layers = MODEL_LAYER_COUNTS.get(
                    cfg.model_id, _get_gguf_layer_count(model_path)
                )

                # Attempt maximum GPU layer offloading first
                current_layers = 99
                # Cap initial active_ctx to 16384 to protect Colab 12.7GB System RAM from OS SIGKILL
                active_ctx = min(cfg.max_context, 16384)

                # KV cache quantization setup (q4_0 default for maximum VRAM savings)
                kv_quant = os.environ.get("MMM_KV_CACHE_QUANT", getattr(cfg, "kv_cache_quant", "q4_0")).lower()
                type_k_val = None
                type_v_val = None
                try:
                    import llama_cpp.llama_cpp as ggml
                    quant_map = {
                        "q4_0": ggml.GGML_TYPE_Q4_0,
                        "q8_0": ggml.GGML_TYPE_Q8_0,
                        "f16": ggml.GGML_TYPE_F16,
                    }
                    type_k_val = quant_map.get(kv_quant)
                    type_v_val = quant_map.get(kv_quant)
                    print(f"⚡ [llama.cpp] KV Cache Quantization: {kv_quant.upper()} (type_k={type_k_val}, type_v={type_v_val})", flush=True)
                except Exception as quant_err:
                    print(f"⚠️ [llama.cpp] KV quant map fallback: {quant_err}", flush=True)

                while True:
                    try:
                        llama_kwargs = {
                            "model_path": model_path,
                            "n_gpu_layers": current_layers,
                            "n_ctx": active_ctx,
                            "n_batch": 512,
                            "n_ubatch": 512,
                            "use_mmap": False,   # Prevents Colab OS System RAM OOM Kernel Crash (SIGKILL 9)
                            "use_mlock": False,  # Prevents RAM lock allocation crash
                            "flash_attn": True,
                            "verbose": True,
                        }
                        if type_k_val is not None:
                            llama_kwargs["type_k"] = type_k_val
                        if type_v_val is not None:
                            llama_kwargs["type_v"] = type_v_val

                        LlamaCppAdapter._llm = Llama(**llama_kwargs)
                        print(
                            f"✅ [llama.cpp] Model initialized successfully (n_ctx={active_ctx}, n_gpu_layers={current_layers})",
                            flush=True,
                        )
                        break
                    except Exception as init_err:
                        err_msg = str(init_err).lower()
                        if "llama_context" in err_msg or "cuda" in err_msg or "out of memory" in err_msg or "failed to" in err_msg or "alloc" in err_msg:
                            base = actual_total_layers if current_layers in (-1, 99) else current_layers
                            if base > 0:
                                new_layers = max(0, base - 5)
                                print(
                                    f"💡 [llama.cpp] VRAM/RAM memory tight. Offloading layers: {base} -> {new_layers}/{actual_total_layers} (ctx={active_ctx})...",
                                    flush=True,
                                )
                                current_layers = new_layers
                            elif active_ctx > 4096:
                                # If layers hit 0, reduce context memory footprint to fit system RAM
                                new_ctx = max(4096, active_ctx // 2)
                                print(
                                    f"⚠️ [llama.cpp] System RAM limits hit. Adjusting active context footprint: {active_ctx} -> {new_ctx}...",
                                    flush=True,
                                )
                                active_ctx = new_ctx
                                current_layers = 99
                            else:
                                raise init_err
                            gc.collect()
                            if torch is not None and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise init_err

                LlamaCppAdapter._current_model_path = model_path
                print(f"✅ [llama.cpp] Model successfully loaded.", flush=True)

            llm = LlamaCppAdapter._llm
            messages = [dict(m) for m in request.messages]

            # Format messages into chat prompt
            try:
                kwargs = {
                    "messages": messages,
                    "max_tokens": cfg.max_new_tokens,
                    "temperature": 0.0,
                }
                if getattr(request, "response_format", None) == "json":
                    kwargs["response_format"] = {"type": "json_object"}

                response = llm.create_chat_completion(**kwargs)
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
