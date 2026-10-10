from __future__ import annotations

"""REG-023: a small successful probe cannot stand in for a larger model request."""

import pytest

from minecraft_mod_ai import llama_exact_context
from minecraft_mod_ai.llama_exact_context import ExactContextOverflow, capacity_safe_payload


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _CountingServer:
    def __init__(self):
        self.token_requests = []
        self.props_calls = 0

    def post(self, url, json):
        assert url.endswith("/chat/completions/input_tokens")
        content = json["messages"][0]["content"]
        self.token_requests.append(content)
        return _Response({"input_tokens": len(content) // 4 + 12})

    def get(self, url):
        assert url.endswith("/props")
        self.props_calls += 1
        return _Response({"default_generation_settings": {"n_ctx": 128}})


def test_small_probe_does_not_certify_larger_context_without_recount(monkeypatch):
    server = _CountingServer()
    llama_exact_context._MANAGED_CONTEXT_CACHE.clear()
    monkeypatch.setattr(llama_exact_context, "_client", lambda _url: server)
    monkeypatch.setattr(
        llama_exact_context,
        "_managed_generation_identity",
        lambda _url: "managed:large-context-regression",
    )
    short = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 16}
    long = {"messages": [{"role": "user", "content": "A" * 600}], "max_tokens": 16}
    assert capacity_safe_payload("http://127.0.0.1:8910/v1", short)["max_tokens"] == 16
    with pytest.raises(ExactContextOverflow, match="input exceeds the active runtime slot"):
        capacity_safe_payload("http://127.0.0.1:8910/v1", long)
    assert server.token_requests == ["ping", "A" * 600]
    assert server.props_calls == 1
