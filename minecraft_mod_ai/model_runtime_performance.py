from __future__ import annotations

import os
import threading
from functools import wraps
from typing import Any, Iterator, Sequence


# Image runtime state is owned by image_runtime_residency. It lives here only as the
# shared state container imported by that contract; no second image-generation wrapper
# is installed from this module.
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

_GIB = 1024**3
_MAX_RERANK_MICROBATCH = 32


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _available_memory_bytes() -> int | None:
    """Return live available system memory without requiring psutil."""

    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(0, int(status.ullAvailPhys))
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return max(0, pages * page_size)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _rerank_microbatch_size(document_count: int) -> int:
    """Choose a bounded batch size from an override or live CPU/RAM capacity."""

    if document_count <= 0:
        return 0
    raw_override = os.environ.get("MMM_RERANK_MICROBATCH", "").strip()
    if raw_override:
        try:
            requested = int(raw_override)
        except ValueError as exc:
            raise ValueError(
                "MMM_RERANK_MICROBATCH must be a positive integer."
            ) from exc
        if requested <= 0:
            raise ValueError("MMM_RERANK_MICROBATCH must be a positive integer.")
        return min(document_count, requested, _MAX_RERANK_MICROBATCH)

    cpu_capacity = min(_MAX_RERANK_MICROBATCH, max(1, (os.cpu_count() or 1) * 2))
    available = _available_memory_bytes()
    ram_capacity = (
        _MAX_RERANK_MICROBATCH
        if available is None
        else max(1, min(_MAX_RERANK_MICROBATCH, available // (2 * _GIB)))
    )
    return min(document_count, cpu_capacity, ram_capacity)


def _length_bucketed_batches(
    rendered: Sequence[str],
    batch_size: int,
) -> Iterator[list[tuple[int, str]]]:
    """Yield stable short-to-long batches while retaining each original index."""

    if batch_size <= 0:
        raise ValueError("Reranker microbatch size must be positive.")
    ordered = sorted(enumerate(rendered), key=lambda item: (len(item[1]), item[0]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def install() -> None:
    """Cache CPU retrieval runtimes; image residency is installed separately."""

    from .model_adapters.embedding import EmbeddingAdapter
    from .model_adapters.reranker import RerankerAdapter

    _install_cached_embedding(EmbeddingAdapter)
    _install_cached_reranker(RerankerAdapter)


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
            "Retrieve the exact approved Minecraft loader and mappings evidence that directly "
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
                yes_id = tokenizer.convert_tokens_to_ids("yes")
                no_id = tokenizer.convert_tokens_to_ids("no")
                values: list[float | None] = [None] * len(rendered)
                batch_size = _rerank_microbatch_size(len(rendered))
                model_device = next(model.parameters()).device
                for batch in _length_bucketed_batches(rendered, batch_size):
                    original_indices = [index for index, _text in batch]
                    batch_rendered = [text for _index, text in batch]
                    inputs = tokenizer(
                        batch_rendered,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_context,
                        return_tensors="pt",
                    ).to(model_device)
                    with torch.inference_mode():
                        logits = model(**inputs).logits[:, -1, [no_id, yes_id]]
                        probabilities = torch.softmax(logits, dim=-1)[:, 1]
                    batch_values = probabilities.detach().cpu().tolist()
                    if len(batch_values) != len(original_indices):
                        raise RuntimeError(
                            "Reranker returned a different number of scores than "
                            "inputs."
                        )
                    for index, value in zip(
                        original_indices,
                        batch_values,
                        strict=True,
                    ):
                        values[index] = float(value)
                if any(value is None for value in values):
                    raise RuntimeError("Reranker did not score every document.")
            return [float(value) for value in values if value is not None]
        except Exception as exc:
            from .model_adapters.base import ModelBackendError

            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc

    cached_score._mmm_cached_reranker_model = True
    cls.score = cached_score
