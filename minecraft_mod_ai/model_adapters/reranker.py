from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..model_runtime_performance import _length_bucketed_batches, _rerank_microbatch_size
from .base import AdapterConfig, ModelBackendError, require_package, torch_dtype


_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    "provided Instruct. Output only yes or no."
)


@dataclass
class _RerankerBackend:
    model: Any
    tokenizer: Any
    lock: threading.RLock = field(default_factory=threading.RLock)


_BACKEND_CACHE_LOCK = threading.RLock()
_BACKEND_CACHE: dict[tuple[str, str, str, str], _RerankerBackend] = {}


def _cache_enabled() -> bool:
    raw = os.environ.get("MMM_CPU_RETRIEVAL_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class RerankerAdapter:
    """Instruction-aware Qwen reranker with native shared backend residency.

    The router creates lightweight adapter objects per request. Keeping the model
    only on an adapter instance therefore causes repeated multi-gigabyte loads. A
    backend is cached by immutable model/runtime configuration and guarded by its
    own lock, so unrelated reranker profiles do not share one global serialization
    point.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self._backend: _RerankerBackend | None = None

    def _backend_key(self) -> tuple[str, str, str, str]:
        return (
            self.config.model_id,
            str(self.config.extra.get("device", "cpu")),
            self.config.torch_dtype,
            str(self.config.extra.get("revision", "")).strip(),
        )

    def _load_backend(self) -> _RerankerBackend:
        require_package("transformers", minimum="4.52.0")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id, device, dtype_name, revision = self._backend_key()
        tokenizer_options: dict[str, Any] = {"padding_side": "left"}
        model_options: dict[str, Any] = {
            "torch_dtype": torch_dtype(dtype_name),
            "device_map": device if device != "cpu" else None,
            "low_cpu_mem_usage": True,
        }
        if revision:
            tokenizer_options["revision"] = revision
            model_options["revision"] = revision

        started = time.monotonic()
        print(
            "retrieval reranker: model load start",
            f"model={model_id}",
            f"device={device}",
            file=sys.stderr,
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_options)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
        if device == "cpu":
            model = model.to("cpu")
        model.eval()
        backend = _RerankerBackend(model=model, tokenizer=tokenizer)
        print(
            "retrieval reranker: model load done",
            f"elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return backend

    def _ensure_backend(self) -> _RerankerBackend:
        if self._backend is not None:
            return self._backend

        key = self._backend_key()
        if not _cache_enabled():
            self._backend = self._load_backend()
            return self._backend

        with _BACKEND_CACHE_LOCK:
            backend = _BACKEND_CACHE.get(key)
            if backend is None:
                backend = self._load_backend()
                _BACKEND_CACHE[key] = backend
            self._backend = backend
            return backend

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str = (
            "Retrieve the exact approved Minecraft loader and mappings evidence that directly "
            "answers the query. Reject other loaders and versions."
        ),
    ) -> list[float]:
        query = query.strip()
        docs = [str(document).strip() for document in documents]
        if not query or not docs or any(not document for document in docs):
            raise ValueError("Reranker query and documents must be non-empty.")

        try:
            import torch

            backend = self._ensure_backend()
            with backend.lock:
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
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ]
                    for prompt in prompts
                ]
                rendered = [
                    backend.tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for message in messages
                ]
                yes_id = backend.tokenizer.convert_tokens_to_ids("yes")
                no_id = backend.tokenizer.convert_tokens_to_ids("no")
                values: list[float | None] = [None] * len(rendered)
                batch_size = _rerank_microbatch_size(len(rendered))
                model_device = next(backend.model.parameters()).device
                started = time.monotonic()
                print(
                    "retrieval reranker: score start",
                    f"documents={len(docs)}",
                    f"microbatch={batch_size}",
                    file=sys.stderr,
                    flush=True,
                )
                for batch in _length_bucketed_batches(rendered, batch_size):
                    original_indices = [index for index, _text in batch]
                    batch_rendered = [text for _index, text in batch]
                    inputs = backend.tokenizer(
                        batch_rendered,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_context,
                        return_tensors="pt",
                    ).to(model_device)
                    with torch.inference_mode():
                        logits = backend.model(**inputs).logits[:, -1, [no_id, yes_id]]
                        probabilities = torch.softmax(logits, dim=-1)[:, 1]
                    batch_values = probabilities.detach().cpu().tolist()
                    if len(batch_values) != len(original_indices):
                        raise RuntimeError(
                            "Reranker returned a different number of scores than inputs."
                        )
                    for index, value in zip(
                        original_indices,
                        batch_values,
                        strict=True,
                    ):
                        values[index] = float(value)
                if any(value is None for value in values):
                    raise RuntimeError("Reranker did not score every document.")
                print(
                    "retrieval reranker: score done",
                    f"documents={len(docs)}",
                    f"elapsed={time.monotonic() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            return [float(value) for value in values if value is not None]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc


RerankerAdapter.score._mmm_cached_reranker_model = True  # type: ignore[attr-defined]
