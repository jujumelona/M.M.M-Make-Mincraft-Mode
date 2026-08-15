from __future__ import annotations

"""Short-window CPU retrieval coalescing for frozen embed/rerank models.

Concurrent singleton requests sharing one resident CPU model are merged into the
model's native batch dimension. Calls that are already native batches execute
directly: queueing them would add a coalescing delay and serialize work that has
nothing left to combine.
"""

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

from .research_perf_common import env_float, env_int

_MARKER = "_mmm_research_cpu_retrieval_performance_v1"

@dataclass
class _BatchRequest:
    key: tuple[Any, ...]
    adapter: Any
    payload: Any
    future: Future[Any]


class _CoalescingBatcher:
    def __init__(self, process: Callable[[list[_BatchRequest]], None], *, name: str) -> None:
        self._process = process
        self._queue: queue.Queue[_BatchRequest] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, request: _BatchRequest) -> Any:
        self._queue.put(request)
        return request.future.result()

    def _run(self) -> None:
        wait_seconds = env_float("MMM_CPU_RETRIEVAL_BATCH_WAIT_MS", 1.5) / 1000.0
        max_requests = env_int("MMM_CPU_RETRIEVAL_BATCH_REQUESTS", 16, minimum=1, maximum=64)
        while True:
            first = self._queue.get()
            batch = [first]
            deadline = time.monotonic() + wait_seconds
            while len(batch) < max_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            groups: dict[tuple[Any, ...], list[_BatchRequest]] = defaultdict(list)
            for item in batch:
                groups[item.key].append(item)
            for group in groups.values():
                try:
                    self._process(group)
                except BaseException as exc:
                    for item in group:
                        if not item.future.done():
                            item.future.set_exception(exc)


def _adapter_key(adapter: Any) -> tuple[Any, ...]:
    config = adapter.config
    return (
        str(config.model_id),
        str(config.extra.get("device", "cpu")),
        str(getattr(config, "torch_dtype", "")),
        str(config.extra.get("revision", "")),
    )


def _install_cpu_retrieval_coalescing() -> None:
    from .model_adapters.embedding import EmbeddingAdapter
    from .model_adapters.reranker import RerankerAdapter

    current_embed = EmbeddingAdapter.embed
    if not getattr(current_embed, _MARKER, False):
        def process_embed(group: list[_BatchRequest]) -> None:
            combined: list[str] = []
            lengths: list[int] = []
            for item in group:
                texts = list(item.payload)
                lengths.append(len(texts))
                combined.extend(texts)
            values = current_embed(group[0].adapter, combined)
            if len(values) != len(combined):
                raise RuntimeError("Embedding batch returned the wrong row count.")
            offset = 0
            for item, length in zip(group, lengths, strict=True):
                item.future.set_result(values[offset : offset + length])
                offset += length

        embed_batcher = _CoalescingBatcher(process_embed, name="mmm-cpu-embed-batcher")

        @wraps(current_embed)
        def embed(self: Any, texts: Sequence[str]) -> list[list[float]]:
            cleaned = tuple(str(text) for text in texts)
            if (
                len(cleaned) != 1
                or threading.current_thread().name == "mmm-cpu-embed-batcher"
            ):
                return current_embed(self, cleaned)
            future: Future[Any] = Future()
            return embed_batcher.submit(
                _BatchRequest(
                    key=_adapter_key(self),
                    adapter=self,
                    payload=cleaned,
                    future=future,
                )
            )

        setattr(embed, _MARKER, True)
        embed.__wrapped__ = current_embed  # type: ignore[attr-defined]
        EmbeddingAdapter.embed = embed

    current_score = RerankerAdapter.score
    if not getattr(current_score, _MARKER, False):
        def process_rerank(group: list[_BatchRequest]) -> None:
            query, instruction = group[0].payload[0], group[0].payload[2]
            documents: list[str] = []
            lengths: list[int] = []
            for item in group:
                docs = list(item.payload[1])
                lengths.append(len(docs))
                documents.extend(docs)
            values = current_score(
                group[0].adapter,
                query,
                documents,
                instruction=instruction,
            )
            if len(values) != len(documents):
                raise RuntimeError("Reranker batch returned the wrong score count.")
            offset = 0
            for item, length in zip(group, lengths, strict=True):
                item.future.set_result(values[offset : offset + length])
                offset += length

        rerank_batcher = _CoalescingBatcher(process_rerank, name="mmm-cpu-rerank-batcher")

        @wraps(current_score)
        def score(
            self: Any,
            query: str,
            documents: Sequence[str],
            *,
            instruction: str = (
                "Retrieve the exact approved Minecraft loader and mappings evidence that directly "
                "answers the query. Reject other loaders and versions."
            ),
        ) -> list[float]:
            docs = tuple(str(document) for document in documents)
            if (
                len(docs) != 1
                or threading.current_thread().name == "mmm-cpu-rerank-batcher"
            ):
                return current_score(self, query, docs, instruction=instruction)
            future: Future[Any] = Future()
            return rerank_batcher.submit(
                _BatchRequest(
                    key=(*_adapter_key(self), str(query), str(instruction)),
                    adapter=self,
                    payload=(str(query), docs, str(instruction)),
                    future=future,
                )
            )

        setattr(score, _MARKER, True)
        score.__wrapped__ = current_score  # type: ignore[attr-defined]
        RerankerAdapter.score = score


def harden() -> None:
    _install_cpu_retrieval_coalescing()


__all__ = ["harden"]
