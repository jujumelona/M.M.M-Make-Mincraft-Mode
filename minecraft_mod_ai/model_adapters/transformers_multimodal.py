from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

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


def _is_qwen35(model_id: str) -> bool:
    return "qwen3.5" in model_id.lower()


def _qwen35_fast_path() -> None:
    """Try binding Qwen3.5 optional CUDA kernels if installed; fallback silently if absent."""
    try:
        modeling = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        causal_conv = importlib.import_module("causal_conv1d")
        gated_delta = importlib.import_module("fla.ops.gated_delta_rule")
        fla_modules = importlib.import_module("fla.modules")
        bindings: dict[str, Any] = {
            "causal_conv1d_fn": getattr(causal_conv, "causal_conv1d_fn", None),
            "causal_conv1d_update": getattr(causal_conv, "causal_conv1d_update", None),
            "chunk_gated_delta_rule": getattr(gated_delta, "chunk_gated_delta_rule", None),
            "fused_recurrent_gated_delta_rule": getattr(gated_delta, "fused_recurrent_gated_delta_rule", None),
            "FusedRMSNormGated": getattr(fla_modules, "FusedRMSNormGated", None),
        }
        all_bound = True
        for name, value in bindings.items():
            if value is not None:
                setattr(modeling, name, value)
            else:
                all_bound = False
        if all_bound:
            setattr(modeling, "is_fast_path_available", True)
    except Exception:
        pass


def _is_cuda_oom(torch_module: Any, exc: BaseException) -> bool:
    candidates = (
        getattr(torch_module, "OutOfMemoryError", None),
        getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None),
    )
    return any(
        isinstance(error_type, type) and isinstance(exc, error_type)
        for error_type in candidates
    )


def _normalize_messages(
    messages: list[Mapping[str, Any]], media_paths: tuple[Path, ...]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = [
        {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
        for msg in messages
    ]
    if not normalized:
        raise ModelConfigurationError(
            "Transformers multimodal generation requires a user message payload."
        )
    if not media_paths:
        return normalized

    user_text = normalized[-1]["content"]
    content: list[dict[str, Any]] = []
    for path in media_paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ModelConfigurationError(f"Media file does not exist: {resolved}")
        content.append({"type": "image", "image": str(resolved)})
    content.append({"type": "text", "text": user_text})
    normalized[-1]["content"] = content
    return normalized


class TransformersMultimodalAdapter(ModelAdapter):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._processor: Any = None
        self._model: Any = None
        self._session_active = False

    @contextmanager
    def generation_session(self):
        """Pin one lazy-loaded processor/model pair for paginated generation."""

        if self._session_active:
            raise ModelConfigurationError(
                "Transformers multimodal generation session is already active."
            )
        self._session_active = True
        try:
            yield
        finally:
            self._session_active = False
            self.close()

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        input_tokens: int | None = None
        torch_module: Any | None = None
        phase = "dependency_check"
        try:
            require_package(
                "transformers",
                minimum="4.48.0",
                maximum_exclusive="5.0.0",
            )
            require_package("accelerate", minimum="1.0.0")
            if _is_qwen35(cfg.model_id):
                require_package(
                    "flash-linear-attention",
                    minimum="0.5.1",
                    maximum_exclusive="0.6",
                )
                require_package("causal-conv1d", minimum="1.4.0")
            import torch
            import transformers

            AutoProcessor = getattr(transformers, "AutoProcessor", None)
            AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
            getattr(transformers, "AutoModelForCausalLM", None)
            getattr(transformers, "AutoModel", None)
            getattr(transformers, "AutoModelForMultimodalLM", None)

            torch_module = torch
            if _is_qwen35(cfg.model_id):
                _qwen35_fast_path()
            phase = "processor_load"
            if self._processor is None:
                try:
                    self._processor = AutoProcessor.from_pretrained(
                        cfg.model_id,
                        trust_remote_code=False,
                    )
                except Exception:
                    self._processor = AutoTokenizer.from_pretrained(
                        cfg.model_id,
                        trust_remote_code=False,
                    )
            processor = self._processor
            phase = "input_render"
            messages = _normalize_messages(list(request.messages), request.media_paths)
            
            tokenizer_or_proc = getattr(processor, "tokenizer", processor)
            extra_template_kwargs: dict[str, Any] = {}
            if _is_qwen35(cfg.model_id) and request.response_format == "json":
                extra_template_kwargs["enable_thinking"] = False
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    **extra_template_kwargs,
                )
            except Exception:
                rendered = tokenizer_or_proc.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **extra_template_kwargs,
                )
                inputs = tokenizer_or_proc(rendered, return_tensors="pt")
            input_tokens = int(inputs["input_ids"].shape[-1])
            requested_tokens = input_tokens + cfg.max_new_tokens
            if requested_tokens > cfg.max_context:
                raise ModelConfigurationError(
                    "Rendered multimodal request exceeds the model context: "
                    f"{input_tokens} input + {cfg.max_new_tokens} reserved output "
                    f"> max_context={cfg.max_context}."
                )
            if (
                cfg.max_input_tokens > 0
                and input_tokens > cfg.max_input_tokens
            ):
                raise ModelConfigurationError(
                    "Rendered multimodal page exceeds this hardware profile's "
                    "per-call input budget: "
                    f"{input_tokens} input tokens > "
                    f"max_input_tokens={cfg.max_input_tokens}. Split or compact "
                    "this stage into additional pages; this is not a project-wide "
                    "content limit."
                )

            # Validate the fully rendered request before loading multi-gigabyte
            # model weights.  An oversized prompt is a configuration error, not
            # a reason to spend GPU memory and tens of seconds loading a model.
            if self._model is None:
                phase = "model_load"
                preflight_cuda(cfg)
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
                candidate_names = [
                    "AutoModelForMultimodalLM",
                    "AutoModelForVision2Seq",
                    "AutoModelForCausalLM",
                    "AutoModelForSeq2SeqLM",
                    "AutoModel",
                ]
                model_loaded = None
                for trust_remote in (False, True):
                    kwargs["trust_remote_code"] = trust_remote
                    for name in candidate_names:
                        auto_cls = getattr(transformers, name, None)
                        if auto_cls is None:
                            continue
                        try:
                            model_loaded = auto_cls.from_pretrained(cfg.model_id, **kwargs)
                            break
                        except Exception:
                            continue
                    if model_loaded is not None:
                        break
                if model_loaded is None:
                    fallback_cls = getattr(transformers, "AutoModelForMultimodalLM", None) or getattr(transformers, "AutoModelForCausalLM", None)
                    if fallback_cls is not None:
                        model_loaded = fallback_cls.from_pretrained(cfg.model_id, trust_remote_code=True, **{k: v for k, v in kwargs.items() if k != "trust_remote_code"})
                self._model = model_loaded
            model = self._model
            device = next(model.parameters()).device
            phase = "input_move"
            inputs = inputs.to(device)
            phase = "generate"
            import re
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
                        msg = str(ve)
                        if "are not used by the model:" in msg:
                            unused_keys = re.findall(r"'([^']+)'", msg)
                            popped = False
                            for key in unused_keys:
                                if key in gen_inputs:
                                    gen_inputs.pop(key, None)
                                    popped = True
                            if popped:
                                continue
                        raise
            generated = output[:, input_tokens:]
            phase = "decode"
            return processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
        except ModelBackendError:
            raise
        except Exception as exc:
            cause: BaseException | str = exc
            if (
                torch_module is not None
                and input_tokens is not None
                and _is_cuda_oom(torch_module, exc)
            ):
                cause = (
                    "CUDA out of memory during multimodal "
                    f"phase={phase} "
                    f"(input_tokens={input_tokens}, "
                    f"max_new_tokens={cfg.max_new_tokens}, "
                    f"max_input_tokens={cfg.max_input_tokens}, "
                    "attention_backend=sdpa). "
                    f"Backend detail: {exc}"
                )
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=cause
            ) from exc
        finally:
            if not self._session_active:
                self.close()

    def close(self) -> None:
        model = self._model
        processor = self._processor
        self._model = None
        self._processor = None
        if model is not None:
            del model
        if processor is not None:
            del processor
        super().close()
