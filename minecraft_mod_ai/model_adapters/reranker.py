from __future__ import annotations

from typing import Sequence

from .base import AdapterConfig, ModelBackendError, require_package, torch_dtype


_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the "
    "provided Instruct. Output only yes or no."
)


class RerankerAdapter:
    """Instruction-aware Qwen3 reranker using the documented yes/no logits."""

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str = (
            "Retrieve Minecraft Fabric 1.20.1 and Yarn 1.20.1 evidence that directly "
            "answers the query. Reject other loaders and versions."
        ),
    ) -> list[float]:
        query = query.strip()
        docs = [str(document).strip() for document in documents]
        if not query or not docs or any(not document for document in docs):
            raise ValueError("Reranker query and documents must be non-empty.")
        try:
            require_package("transformers", minimum="4.51.0")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = str(self.config.extra.get("device", "cpu"))
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
            yes_id = tokenizer.convert_tokens_to_ids("yes")
            no_id = tokenizer.convert_tokens_to_ids("no")
            with torch.inference_mode():
                logits = model(**inputs).logits[:, -1, [no_id, yes_id]]
                probabilities = torch.softmax(logits, dim=-1)[:, 1]
            return [float(value) for value in probabilities.detach().cpu().tolist()]
        except Exception as exc:
            raise ModelBackendError(
                role=self.config.role,
                model_id=self.config.model_id,
                cause=exc,
            ) from exc
