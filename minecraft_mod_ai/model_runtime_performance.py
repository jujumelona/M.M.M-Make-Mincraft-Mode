from __future__ import annotations

import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Sequence


_IMAGE_LOCK = threading.RLock()
_IMAGE_PIPELINE: Any | None = None
_IMAGE_PIPELINE_KEY: tuple[Any, ...] | None = None
_EMBED_LOCK = threading.RLock()
_EMBED_MODEL: Any | None = None
_EMBED_KEY: tuple[Any, ...] | None = None
_RERANK_LOCK = threading.RLock()
_RERANK_TOKENIZER: Any | None = None
_RERANK_MODEL: Any | None = None
_RERANK_KEY: tuple[Any, ...] | None = None


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def install() -> None:
    """Install quality-neutral reuse for expensive image/retrieval runtimes.

    Native llama-server exclusively owns GGUF model residency and GPU handoff. This
    contract caches only diffusion, embedding and reranker runtimes and prevents
    independent local GPU work from interleaving inside one asset shard.
    """

    from . import complete_orchestrator_services as services
    from . import model_router as router_module
    from .model_adapters import base as base_module
    from .model_adapters.embedding import EmbeddingAdapter
    from .model_adapters.image_diffusion import ImageDiffusionAdapter
    from .model_adapters.reranker import RerankerAdapter

    _install_cached_image_runtime(ImageDiffusionAdapter, base_module)
    _install_cached_embedding(EmbeddingAdapter)
    _install_cached_reranker(RerankerAdapter)
    _install_asset_gpu_session(services, router_module, base_module)


def _install_cached_image_runtime(cls: Any, base_module: Any) -> None:
    original = cls.generate_image
    if getattr(original, "_mmm_cached_image_pipeline", False):
        return

    @wraps(original)
    def cached_generate_image(
        self: Any,
        *,
        prompt: str,
        output_path: Path,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Path:
        global _IMAGE_PIPELINE, _IMAGE_PIPELINE_KEY

        cfg = self.config
        try:
            base_module.require_package("diffusers", minimum="0.20.0")
            base_module.require_package("transformers", minimum="4.52.0")
            base_module.require_package("accelerate", minimum="1.0.0")
            if not prompt.strip():
                raise base_module.ModelConfigurationError("Image prompt is empty.")
            if (
                width % 16
                or height % 16
                or not (256 <= width <= 1024 and 256 <= height <= 1024)
            ):
                raise base_module.ModelConfigurationError(
                    "Image dimensions must be 256-1024 and divisible by 16."
                )

            # Native llama-server eviction is owned by the GPU handoff contracts.
            # This layer only manages the diffusion runtime, so it must not mutate
            # text-model state or duplicate CUDA/model teardown work.
            base_module.preflight_cuda(cfg)

            import torch
            from diffusers import DiffusionPipeline

            cache_enabled = bool(cfg.cpu_offload) and _env_bool(
                "MMM_IMAGE_PIPELINE_CACHE",
                True,
            )
            key = (
                cfg.model_id,
                cfg.torch_dtype,
                bool(cfg.cpu_offload),
            )

            with _IMAGE_LOCK:
                pipeline = None
                if cache_enabled and _IMAGE_PIPELINE_KEY == key:
                    pipeline = _IMAGE_PIPELINE
                if pipeline is None:
                    if _IMAGE_PIPELINE is not None:
                        try:
                            del _IMAGE_PIPELINE
                        except Exception:
                            pass
                        _IMAGE_PIPELINE = None
                        _IMAGE_PIPELINE_KEY = None
                        base_module._release_cuda()
                    pipeline = DiffusionPipeline.from_pretrained(
                        cfg.model_id,
                        torch_dtype=base_module.torch_dtype(cfg.torch_dtype),
                        trust_remote_code=False,
                    )
                    if cfg.cpu_offload:
                        pipeline.enable_model_cpu_offload()
                    else:
                        pipeline.to("cuda")
                    set_progress = getattr(pipeline, "set_progress_bar_config", None)
                    if callable(set_progress):
                        set_progress(disable=True)
                    if cache_enabled:
                        _IMAGE_PIPELINE = pipeline
                        _IMAGE_PIPELINE_KEY = key

                generator = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    result = pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        generator=generator,
                        num_inference_steps=4,
                        guidance_scale=1.0,
                    )
                output = output_path.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(output)

                if not cache_enabled:
                    del pipeline
                    base_module._release_cuda()
                return output
        except base_module.ModelBackendError:
            raise
        except Exception as exc:
            # Do not retain a partially failed diffusion runtime.
            with _IMAGE_LOCK:
                if _IMAGE_PIPELINE_KEY == (
                    cfg.model_id,
                    cfg.torch_dtype,
                    bool(cfg.cpu_offload),
                ):
                    _IMAGE_PIPELINE = None
                    _IMAGE_PIPELINE_KEY = None
            base_module._release_cuda()
            raise base_module.ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc

    cached_generate_image._mmm_cached_image_pipeline = True
    cls.generate_image = cached_generate_image


def _install_cached_embedding(cls: Any) -> None:
    original = cls.embed
    if getattr(original, "_mmm_cached_embedding_model", False):
        return

    @wraps(original)
    def cached_embed(self: Any, texts: Sequence[str]) -> list[list[float]]:
        global _EMBED_MODEL, _EMBED_KEY

        cleaned = [str(text).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("Embedding input must contain non-empty strings.")
        try:
            from .model_adapters.base import require_package

            require_package("sentence-transformers", minimum="3.0.0")
            from sentence_transformers import SentenceTransformer

            device = str(self.config.extra.get("device", "cpu"))
            dimensions = int(self.config.extra.get("dimensions", 512))
            revision = str(self.config.extra.get("revision", "")).strip()
            key = (self.config.model_id, device, revision)
            cache_enabled = _env_bool("MMM_CPU_RETRIEVAL_CACHE", True)

            with _EMBED_LOCK:
                model = _EMBED_MODEL if cache_enabled and _EMBED_KEY == key else None
                if model is None:
                    options: dict[str, object] = {
                        "device": device,
                        "trust_remote_code": False,
                    }
                    if revision:
                        options["revision"] = revision
                    model = SentenceTransformer(self.config.model_id, **options)
                    if cache_enabled:
                        _EMBED_MODEL = model
                        _EMBED_KEY = key
                vectors = model.encode(
                    cleaned,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    truncate_dim=dimensions,
                    show_progress_bar=False,
                )
            return [[float(value) for value in row] for row in vectors]
        except Exception as exc:
            from .model_adapters.base import ModelBackendError

            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc

    cached_embed._mmm_cached_embedding_model = True
    cls.embed = cached_embed


def _install_cached_reranker(cls: Any) -> None:
    original = cls.score
    if getattr(original, "_mmm_cached_reranker_model", False):
        return

    @wraps(original)
    def cached_score(
        self: Any,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str = (
            "Retrieve Minecraft Fabric 1.20.1 and Yarn 1.20.1 evidence that directly "
            "answers the query. Reject other loaders and versions."
        ),
    ) -> list[float]:
        global _RERANK_TOKENIZER, _RERANK_MODEL, _RERANK_KEY

        query = query.strip()
        docs = [str(document).strip() for document in documents]
        if not query or not docs or any(not document for document in docs):
            raise ValueError("Reranker query and documents must be non-empty.")
        try:
            from .model_adapters.base import require_package, torch_dtype

            require_package("transformers", minimum="4.52.0")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = str(self.config.extra.get("device", "cpu"))
            revision = str(self.config.extra.get("revision", "")).strip()
            key = (
                self.config.model_id,
                device,
                self.config.torch_dtype,
                revision,
            )
            cache_enabled = _env_bool("MMM_CPU_RETRIEVAL_CACHE", True)

            with _RERANK_LOCK:
                tokenizer = (
                    _RERANK_TOKENIZER
                    if cache_enabled and _RERANK_KEY == key
                    else None
                )
                model = (
                    _RERANK_MODEL
                    if cache_enabled and _RERANK_KEY == key
                    else None
                )
                if tokenizer is None or model is None:
                    tokenizer_kwargs: dict[str, Any] = {"padding_side": "left"}
                    model_kwargs: dict[str, Any] = {
                        "torch_dtype": torch_dtype(self.config.torch_dtype),
                        "device_map": (device if device != "cpu" else None),
                        "low_cpu_mem_usage": True,
                    }
                    if revision:
                        tokenizer_kwargs["revision"] = revision
                        model_kwargs["revision"] = revision
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_id,
                        **tokenizer_kwargs,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_id,
                        **model_kwargs,
                    )
                    if device == "cpu":
                        model = model.to("cpu")
                    model.eval()
                    if cache_enabled:
                        _RERANK_TOKENIZER = tokenizer
                        _RERANK_MODEL = model
                        _RERANK_KEY = key

                prompts = [
                    (
                        f"<Instruct>: {instruction}\n"
                        f"<Query>: {query}\n"
                        f"<Document>: {document}"
                    )
                    for document in docs
                ]
                messages = [
                    [
                        {
                            "role": "system",
                            "content": (
                                "Judge whether the Document meets the requirements "
                                "based on the Query and the provided Instruct. "
                                "Output only yes or no."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    for prompt in prompts
                ]
                rendered = [
                    tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for message in messages
                ]
                inputs = tokenizer(
                    rendered,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_context,
                    return_tensors="pt",
                ).to(next(model.parameters()).device)
                yes_id = tokenizer.convert_tokens_to_ids("yes")
                no_id = tokenizer.convert_tokens_to_ids("no")
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, [no_id, yes_id]]
                    probabilities = torch.softmax(logits, dim=-1)[:, 1]
                values = probabilities.detach().cpu().tolist()
            return [float(value) for value in values]
        except Exception as exc:
            from .model_adapters.base import ModelBackendError

            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc

    cached_score._mmm_cached_reranker_model = True
    cls.score = cached_score


def _install_asset_gpu_session(
    services: Any,
    router_module: Any,
    base_module: Any,
) -> None:
    original = services.generate_assets
    if getattr(original, "_mmm_image_gpu_session", False):
        return

    @wraps(original)
    def image_gpu_session(router: Any, *args: Any, **kwargs: Any):
        registry = getattr(router, "registry", None)
        profile = getattr(router, "profile", None)
        if registry is None or profile is None:
            # Unit-test/lightweight routers intentionally implement only
            # generate_image(). They have no model registry and must keep the
            # original deterministic asset-generation contract.
            return original(router, *args, **kwargs)
        config = registry.role(profile, "image_generator")
        local_exclusive = (
            config.provider == "local"
            and config.adapter == "image_diffusion"
            and config.exclusive_gpu
        )
        if not local_exclusive:
            return original(router, *args, **kwargs)

        # One asset shard owns the GPU as a unit. Individual diffusion calls use the
        # same RLock recursively, so a waiting local LLM cannot slip between overview
        # and detail tiles and force both large runtimes to reload repeatedly.
        with router_module._GPU_EXCLUSIVE_LOCK:
            try:
                return original(router, *args, **kwargs)
            finally:
                base_module._release_cuda()

    image_gpu_session._mmm_image_gpu_session = True
    services.generate_assets = image_gpu_session
