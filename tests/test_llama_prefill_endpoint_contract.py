from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _assistant_prefill_apply_template_url,
)


@pytest.mark.parametrize(
    ("server_url", "expected"),
    [
        ("http://127.0.0.1:8910/v1", "http://127.0.0.1:8910/apply-template"),
        ("http://127.0.0.1:8910/v1/", "http://127.0.0.1:8910/apply-template"),
        ("http://127.0.0.1:8910", "http://127.0.0.1:8910/apply-template"),
    ],
)
def test_apply_template_endpoint_is_server_root(server_url: str, expected: str) -> None:
    assert _assistant_prefill_apply_template_url(server_url) == expected
