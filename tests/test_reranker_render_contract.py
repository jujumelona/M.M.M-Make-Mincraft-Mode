from __future__ import annotations

from minecraft_mod_ai.model_adapters.reranker import _SYSTEM, _render_rerank_input


class _Tokenizer:
    def __init__(self) -> None:
        self.message = None

    def apply_chat_template(self, message, *, tokenize, add_generation_prompt):
        self.message = message
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered"


def test_render_rerank_input_preserves_chat_contract() -> None:
    tokenizer = _Tokenizer()

    rendered = _render_rerank_input(
        tokenizer,
        query="query",
        instruction="instruction",
        document="document",
    )

    assert rendered == "rendered"
    assert tokenizer.message == [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": "<Instruct>: instruction\n<Query>: query\n<Document>: document",
        },
    ]
