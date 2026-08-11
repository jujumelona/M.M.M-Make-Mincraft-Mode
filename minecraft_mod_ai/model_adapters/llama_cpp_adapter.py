"""llama.cpp GGUF inference adapter."""
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
    """Read actual block_count / layer_count from a GGUF header when possible."""
    import struct

    try:
        with open(model_path, "rb") as f:
            header = f.read(4 * 1024 * 1024)
            pos = header.find(b".block_count")
            if pos != -1:
                for offset in range(
                    pos + len(b".block_count"),
                    min(len(header) - 8, pos + len(b".block_count") + 16),
                ):
                    try:
                        val_type, val = struct.unpack("<II", header[offset : offset + 8])
                        if val_type in (4, 5, 8, 9, 10) and 4 <= val <= 128:
                            return val
                    except Exception:
                        pass
    except Exception:
        pass
    return 32


def _verbose_enabled() -> bool:
    raw = os.environ.get("MMM_LLAMA_CPP_VERBOSE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class LlamaCppAdapter(ModelAdapter):
    """llama-cpp-python GGUF engine with GPU layer offloading."""

    _llm: Any = None
    _current_model_path: str | None = None
    _reported_server_url: str | None = None

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._llm_instance: Any = None

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config

        server_url = os.environ.get(
            "LLAMA_SERVER_URL", "http://localhost:8910/v1"
        ).rstrip("/")
        try:
            import httpx

            check_resp = httpx.get(f"{server_url}/models", timeout=0.5)
            if check_resp.status_code == 200:
                if LlamaCppAdapter._reported_server_url != server_url:
                    print("llama server: connected", server_url, flush=True)
                    LlamaCppAdapter._reported_server_url = server_url
                messages = [dict(m) for m in request.messages]
                payload: dict[str, Any] = {
                    "model": "local",
                    "messages": messages,
                    "max_tokens": cfg.max_new_tokens,
                    "temperature": 0.0,
                }
                if getattr(request, "response_format", None) == "json":
                    payload["response_format"] = {"type": "json_object"}

                resp = httpx.post(
                    f"{server_url}/chat/completions",
                    json=payload,
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                print(
                    "llama server: request failed",
                    resp.status_code,
                    flush=True,
                )
        except Exception:
            pass

        require_package("llama-cpp-python")
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        try:
            repo_id = cfg.model_id
            filename = cfg.extra.get("gguf_filename", "") or ""

            if os.path.exists(repo_id) and repo_id.endswith(".gguf"):
                model_path = repo_id
            else:
                hub_repo = repo_id
                if "/" not in hub_repo:
                    hub_repo = f"bartowski/{hub_repo}-GGUF"
                elif (
                    not hub_repo.endswith("-GGUF")
                    and "GGUF" not in hub_repo
                    and "gguf" not in hub_repo
                ):
                    repo_name = hub_repo.split("/")[-1]
                    hub_repo = f"bartowski/{repo_name}-GGUF"

                if not filename:
                    from huggingface_hub import list_repo_files

                    repo_files = list_repo_files(hub_repo)
                    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
                    if gguf_files:
                        q4_files = [
                            f
                            for f in gguf_files
                            if "Q4_K_M" in f or "q4_k_m" in f or "Q4_0" in f
                        ]
                        filename = q4_files[0] if q4_files else gguf_files[0]
                    else:
                        filename = f"{hub_repo.split('/')[-1]}.gguf"

                print("llama.cpp: downloading GGUF", hub_repo, filename, flush=True)
                model_path = hf_hub_download(repo_id=hub_repo, filename=filename)

            if (
                LlamaCppAdapter._llm is not None
                and LlamaCppAdapter._current_model_path != model_path
            ):
                del LlamaCppAdapter._llm
                LlamaCppAdapter._llm = None
                LlamaCppAdapter._current_model_path = None
                gc.collect()

            if LlamaCppAdapter._llm is None:
                print("llama.cpp: model loading", model_path, flush=True)

                try:
                    import torch
                except ImportError:
                    torch = None

                actual_total_layers = MODEL_LAYER_COUNTS.get(
                    cfg.model_id, _get_gguf_layer_count(model_path)
                )
                current_layers = 99
                active_ctx = min(cfg.max_context, 16384)

                kv_quant = os.environ.get(
                    "MMM_KV_CACHE_QUANT",
                    getattr(cfg, "kv_cache_quant", "q4_0"),
                ).lower()
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
                    print("llama.cpp: KV cache", kv_quant, flush=True)
                except Exception as quant_err:
                    print("llama.cpp: KV cache fallback", quant_err, flush=True)

                while True:
                    try:
                        llama_kwargs = {
                            "model_path": model_path,
                            "n_gpu_layers": current_layers,
                            "n_ctx": active_ctx,
                            "n_batch": 512,
                            "n_ubatch": 512,
                            "use_mmap": False,
                            "use_mlock": False,
                            "flash_attn": True,
                            "verbose": _verbose_enabled(),
                        }
                        if type_k_val is not None:
                            llama_kwargs["type_k"] = type_k_val
                        if type_v_val is not None:
                            llama_kwargs["type_v"] = type_v_val

                        LlamaCppAdapter._llm = Llama(**llama_kwargs)
                        print(
                            "llama.cpp: model loaded",
                            f"ctx={active_ctx}",
                            f"gpu_layers={current_layers}",
                            flush=True,
                        )
                        break
                    except Exception as init_err:
                        err_msg = str(init_err).lower()
                        memory_error = any(
                            token in err_msg
                            for token in (
                                "llama_context",
                                "cuda",
                                "out of memory",
                                "failed to",
                                "alloc",
                            )
                        )
                        if not memory_error:
                            raise

                        base = (
                            actual_total_layers
                            if current_layers in (-1, 99)
                            else current_layers
                        )
                        if base > 0:
                            new_layers = max(0, base - 5)
                            print(
                                "llama.cpp: retry gpu layers",
                                f"{base}->{new_layers}",
                                f"ctx={active_ctx}",
                                flush=True,
                            )
                            current_layers = new_layers
                        elif active_ctx > 4096:
                            new_ctx = max(4096, active_ctx // 2)
                            print(
                                "llama.cpp: retry context",
                                f"{active_ctx}->{new_ctx}",
                                flush=True,
                            )
                            active_ctx = new_ctx
                            current_layers = 99
                        else:
                            raise init_err

                        gc.collect()
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                LlamaCppAdapter._current_model_path = model_path

            llm = LlamaCppAdapter._llm
            messages = [dict(m) for m in request.messages]

            try:
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "max_tokens": cfg.max_new_tokens,
                    "temperature": 0.0,
                }
                if getattr(request, "response_format", None) == "json":
                    kwargs["response_format"] = {"type": "json_object"}

                response = llm.create_chat_completion(**kwargs)
                return response["choices"][0]["message"]["content"].strip()
            except Exception:
                prompt_text = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                )
                response = llm(
                    prompt_text,
                    max_tokens=cfg.max_new_tokens,
                    temperature=0.0,
                    stop=["<|endoftext|>", "<|im_end|>"],
                )
                return response["choices"][0]["text"].strip()

        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role,
                model_id=cfg.model_id,
                cause=exc,
            ) from exc

    def close(self) -> None:
        if LlamaCppAdapter._llm is not None:
            del LlamaCppAdapter._llm
            LlamaCppAdapter._llm = None
            LlamaCppAdapter._current_model_path = None
            gc.collect()
