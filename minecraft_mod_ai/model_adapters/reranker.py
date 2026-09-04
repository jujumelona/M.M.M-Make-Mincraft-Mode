from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..model_runtime_performance import (
    _length_bucketed_batches,
    _length_prefixed_digest,
    _rerank_microbatch_size,
    _text_digest,
)
from ..model_runtime_performance import (
    _retrieval_cache_enabled as _cache_enabled,
)
from ..model_runtime_performance import (
    _retrieval_result_cache_limit as _result_cache_limit,
)
from ..retrieval_cpu_budget_contract import require_dense_retrieval_device
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
_SCORE_CACHE_LOCK = threading.RLock()
_SCORE_CACHE: OrderedDict[tuple[str, str, str, str, str, str], float] = OrderedDict()


def _request_digest(query: str, instruction: str) -> str:
    return _length_prefixed_digest((query, instruction))


def _render_rerank_input(
    tokenizer: Any,
    *,
    query: str,
    instruction: str,
    document: str,
) -> str:
    """Render one document without retaining prompt/message staging lists."""

    prompt = (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )
    return str(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )


class RerankerAdapter:
    """Instruction-aware Qwen reranker with shared residency and per-document reuse.

    The router creates lightweight adapter objects per request. Keeping the model
    only on an adapter instance therefore causes repeated multi-gigabyte loads. A
    backend is cached by immutable model/runtime configuration and guarded by its
    own lock, so unrelated reranker profiles do not share one global serialization
    point. Scores are cached per query/instruction/document tuple, which prevents
    overlapping candidate batches from repeatedly scoring the same documents.
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

    def _score_cache_keys(
        self,
        query: str,
        instruction: str,
        documents: Sequence[str],
    ) -> dict[str, tuple[str, str, str, str, str, str]]:
        """Build request and document cache commitments once per score call."""

        model_id, device, dtype_name, revision = self._backend_key()
        prefix = (
            model_id,
            device,
            dtype_name,
            revision,
            _request_digest(query, instruction),
        )
        return {
            document: (*prefix, _text_digest(document))
            for document in dict.fromkeys(documents)
        }

    def _load_backend(self) -> _RerankerBackend:
        model_id, device, dtype_name, revision = self._backend_key()
        require_dense_retrieval_device(
            device,
            role=self.config.role,
            model_id=model_id,
            backend="reranker",
        )
        require_package("transformers", minimum="4.52.0")
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        cache_active = _cache_enabled()
        cache_keys = (
            self._score_cache_keys(query, instruction, docs) if cache_active else {}
        )
        scores_by_document: dict[str, float] = {}
        missing: list[str] = []
        seen_missing: set[str] = set()

        if cache_active:
            with _SCORE_CACHE_LOCK:
                for document in docs:
                    cache_key = cache_keys[document]
                    cached = _SCORE_CACHE.get(cache_key)
                    if cached is not None:
                        _SCORE_CACHE.move_to_end(cache_key)
                        scores_by_document[document] = cached
                    elif document not in seen_missing:
                        seen_missing.add(document)
                        missing.append(document)
        else:
            for document in docs:
                if document not in seen_missing:
                    seen_missing.add(document)
                    missing.append(document)

        if not missing:
            print(
                "retrieval reranker: score cache hit",
                f"documents={len(docs)}",
                file=sys.stderr,
                flush=True,
            )
            return [scores_by_document[document] for document in docs]

        try:
            import torch

            backend = self._ensure_backend()
            with backend.lock:
                if cache_active:
                    with _SCORE_CACHE_LOCK:
                        still_missing: list[str] = []
                        for document in missing:
                            cache_key = cache_keys[document]
                            cached = _SCORE_CACHE.get(cache_key)
                            if cached is not None:
                                _SCORE_CACHE.move_to_end(cache_key)
                                scores_by_document[document] = cached
                            else:
                                still_missing.append(document)
                    missing = still_missing
                if not missing:
                    return [scores_by_document[document] for document in docs]
                rendered = [
                    _render_rerank_input(
                        backend.tokenizer,
                        query=query,
                        instruction=instruction,
                        document=document,
                    )
                    for document in missing
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
                    f"missing={len(missing)}",
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
                computed = [float(value) for value in values if value is not None]
                print(
                    "retrieval reranker: score done",
                    f"documents={len(docs)}",
                    f"computed={len(missing)}",
                    f"elapsed={time.monotonic() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

            for document, value in zip(missing, computed, strict=True):
                scores_by_document[document] = value

            if cache_active:
                with _SCORE_CACHE_LOCK:
                    for document, value in zip(missing, computed, strict=True):
                        cache_key = cache_keys[document]
                        _SCORE_CACHE[cache_key] = value
                        _SCORE_CACHE.move_to_end(cache_key)
                    cache_limit = _result_cache_limit()
                    while len(_SCORE_CACHE) > cache_limit:
                        _SCORE_CACHE.popitem(last=False)
            return [scores_by_document[document] for document in docs]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc


RerankerAdapter.score._mmm_cached_reranker_model = True  # type: ignore[attr-defined]
