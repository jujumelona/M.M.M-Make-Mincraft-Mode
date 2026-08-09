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
    """Bind Qwen3.5 CUDA kernels even when Transformers cached them as absent.

    Colab can import Transformers before the setup cell installs optional CUDA
    packages.  Older Qwen3.5 modeling modules then retain ``None`` globals for
    the lifetime of the process and silently instantiate the memory-heavy torch
    reference path.  Import the installed kernels directly and replace those
    cached globals before any model layers or checkpoint weights are created.
    """

    install_hint = (
        "Install flash-linear-attention[cuda,conv1d] in the GPU runtime, then "
        "restart the runtime and rerun setup before planning. Refusing the "
        "memory-heavy Qwen3.5 PyTorch fallback."
    )
    try:
        modeling = importlib.import_module(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
        causal_conv = importlib.import_module("causal_conv1d")
        gated_delta = importlib.import_module("fla.ops.gated_delta_rule")
        bindings: dict[str, Any] = {
            "causal_conv1d_fn": getattr(causal_conv, "causal_conv1d_fn"),
            "causal_conv1d_update": getattr(
                causal_conv, "causal_conv1d_update"
            ),
            "chunk_gated_delta_rule": getattr(
                gated_delta, "chunk_gated_delta_rule"
            ),
            "fused_recurrent_gated_delta_rule": getattr(
                gated_delta, "fused_recurrent_gated_delta_rule"
            ),
        }
        # Transformers releases that expose this global select FLA's fused
        # gated RMSNorm at Qwen layer construction time. Bind it from the same
        # verified installation instead of retaining a stale ``None`` value.
        if hasattr(modeling, "FusedRMSNormGated"):
            fla_modules = importlib.import_module("fla.modules")
            bindings["FusedRMSNormGated"] = getattr(
                fla_modules, "FusedRMSNormGated"
            )
    except Exception as exc:
        raise ModelConfigurationError(
            f"Qwen3.5 fast CUDA kernels are unavailable: {exc}. {install_hint}"
        ) from exc

    invalid = [name for name, value in bindings.items() if not callable(value)]
    if invalid:
        raise ModelConfigurationError(
            "Qwen3.5 fast CUDA kernel bindings are not callable: "
            + ", ".join(sorted(invalid))
            + ". "
            + install_hint
        )
    for name, value in bindings.items():
        setattr(modeling, name, value)
    if hasattr(modeling, "is_fast_path_available"):
        modeling.is_fast_path_available = True

    rebound = [
        name
        for name, value in bindings.items()
        if getattr(modeling, name, None) is not value
    ]
    if rebound:
        raise ModelConfigurationError(
            "Transformers rejected Qwen3.5 fast CUDA kernel bindings: "
            + ", ".join(sorted(rebound))
            + ". "
            + install_hint
        )


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
    normalized = [dict(message) for message in messages]
    if not media_paths:
        return normalized
    if not normalized or normalized[-1].get("role") != "user":
        raise ModelConfigurationError("Media can only be attached to a final user message.")
    user_text = normalized[-1].get("content", "")
    if not isinstance(user_text, str):
        raise ModelConfigurationError("The final user message must contain text.")
    content: list[dict[str, str]] = []
    for path in media_paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ModelConfigurationError(f"Media file does not exist: {resolved}")
        content.append({"type": "image", "url": resolved.as_uri()})
    content.append({"type": "text", "text": user_text})
    normalized[-1] = {"role": "user", "content": content}
    return normalized


class TransformersMultimodalAdapter(ModelAdapter):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._processor: Any | None = None
        self._model: Any | None = None
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
            yield self
        finally:
            self._session_active = False
            self.close()

    def generate(self, request: GenerationRequest) -> str:
        cfg = self.config
        input_tokens: int | None = None
        torch_module: Any | None = None
        phase = "dependency_check"
        try:
            # Qwen3.5's verified fast-path globals and the current ``dtype``
            # loader keyword are pinned to the 5.14 runtime contract.
            require_package(
                "transformers",
                minimum="4.48.0",
                maximum_exclusive="5.0.0",
            )
            require_package("accelerate", minimum="1.0.0")
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoModelForConditionalGeneration, AutoModel

            torch_module = torch
            if _is_qwen35(cfg.model_id):
                require_package(
                    "flash-linear-attention",
                    minimum="0.5.1",
                    maximum_exclusive="0.6",
                )
                require_package("causal-conv1d", minimum="1.4.0")
                _qwen35_fast_path()
            phase = "processor_load"
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    cfg.model_id,
                    trust_remote_code=False,
                )
            processor = self._processor
            phase = "input_render"
            messages = _normalize_messages(list(request.messages), request.media_paths)
            template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            # Qwen3.5 emits a separate <think> response by default.  That is
            # useful for prose, but a structured planner response must begin
            # with its contract JSON so a reasoning draft cannot be mistaken
            # for the actual plan.  The flag is a documented Qwen3.5 chat
            # template option, so do not pass it to unrelated models.
            if (
                request.response_format == "json"
                and cfg.model_id.lower().startswith("qwen/qwen3.5")
            ):
                template_kwargs["enable_thinking"] = False
            inputs = processor.apply_chat_template(
                messages,
                **template_kwargs,
            )
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
                qconfig = quantization_config(cfg)
                if qconfig is not None:
                    kwargs["quantization_config"] = qconfig
                model_loaded = None
                for auto_cls in (AutoModelForVision2Seq, AutoModelForConditionalGeneration, AutoModelForCausalLM, AutoModel):
                    try:
                        model_loaded = auto_cls.from_pretrained(cfg.model_id, **kwargs)
                        break
                    except Exception:
                        continue
                if model_loaded is None:
                    model_loaded = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
                self._model = model_loaded
            model = self._model
            device = next(model.parameters()).device
            phase = "input_move"
            inputs = inputs.to(device)
            phase = "generate"
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                )
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
