from __future__ import annotations

import sys
import threading
import time
from typing import Any, Sequence

from .base import AdapterConfig, ModelBackendError, require_package, torch_dtype


_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    "provided Instruct. Output only yes or no."
)


class RerankerAdapter:
    """Instruction-aware Qwen3 reranker using the documented yes/no logits.

    Model/tokenizer construction is lazy and resident for this adapter's lifetime;
    repeated tool selection and RAG queries must not reload the same CPU model.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure_backend(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        require_package("transformers", minimum="4.52.0")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = str(self.config.extra.get("device", "cpu"))
        started = time.monotonic()
        print(
            "retrieval reranker: model load start",
            f"model={self.config.model_id}",
            f"device={device}",
            file=sys.stderr,
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype(self.config.torch_dtype),
            device_map=(device if device != "cpu" else None),
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to("cpu")
        self._tokenizer = tokenizer
        self._model = model
        print(
            "retrieval reranker: model load done",
            f"elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return model, tokenizer

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
            require_package("transformers", minimum="4.52.0")
            import torch

            with self._lock:
                model, tokenizer = self._ensure_backend()
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
                input_tokens = int(inputs["input_ids"].shape[-1])
                started = time.monotonic()
                print(
                    "retrieval reranker: score start",
                    f"documents={len(docs)}",
                    f"padded_tokens={input_tokens}",
                    file=sys.stderr,
                    flush=True,
                )
                yes_id = tokenizer.convert_tokens_to_ids("yes")
                no_id = tokenizer.convert_tokens_to_ids("no")
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, [no_id, yes_id]]
                    probabilities = torch.softmax(logits, dim=-1)[:, 1]
                print(
                    "retrieval reranker: score done",
                    f"documents={len(docs)}",
                    f"elapsed={time.monotonic() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            return [float(value) for value in probabilities.detach().cpu().tolist()]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc
