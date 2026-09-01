from __future__ import annotations

from minecraft_mod_ai.llama_exact_context import (
    capacity_safe_payload,
    live_context_accounting,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self):
        self.props_calls = 0
        self.token_calls = 0

    def post(self, url, json):
        self.token_calls += 1
        assert url.endswith("/v1/chat/completions/input_tokens")
        assert json["messages"][0]["content"] == "hello"
        return _Response({"object": "response.input_tokens", "input_tokens": 37})

    def get(self, url):
        self.props_calls += 1
        assert url.endswith("/props")
        return _Response({"default_generation_settings": {"n_ctx": 128}})


def _uncached(monkeypatch, client):
    import minecraft_mod_ai.llama_exact_context as module

    monkeypatch.setattr(module, "_client", lambda _url: client)
    monkeypatch.setattr(module, "_managed_generation_identity", lambda _url: "")


def test_live_accounting_uses_server_tokenizer_and_slot(monkeypatch):
    client = _Client()
    _uncached(monkeypatch, client)
    accounting = live_context_accounting(
        "http://127.0.0.1:8910/v1",
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 999},
    )
    assert accounting.input_tokens == 37
    assert accounting.context_tokens == 128
    assert accounting.remaining_tokens == 91


def test_output_allowance_is_only_physical_remaining_context(monkeypatch):
    client = _Client()
    _uncached(monkeypatch, client)
    payload = capacity_safe_payload(
        "http://127.0.0.1:8910/v1",
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 999},
    )
    assert payload["max_tokens"] == 91


def test_no_arbitrary_output_shrink_when_requested_output_physically_fits(monkeypatch):
    client = _Client()
    _uncached(monkeypatch, client)
    payload = capacity_safe_payload(
        "http://127.0.0.1:8910/v1",
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 80},
    )
    assert payload["max_tokens"] == 80


def test_managed_generation_reuses_props_but_recounts_each_payload(monkeypatch):
    import minecraft_mod_ai.llama_exact_context as module

    client = _Client()
    module._MANAGED_CONTEXT_CACHE.clear()
    monkeypatch.setattr(module, "_client", lambda _url: client)
    monkeypatch.setattr(
        module, "_managed_generation_identity", lambda _url: "managed:generation-a"
    )
    for _ in range(3):
        live_context_accounting(
            "http://127.0.0.1:8910/v1",
            {"messages": [{"role": "user", "content": "hello"}]},
        )
    assert client.token_calls == 3
    assert client.props_calls == 1


def test_managed_restart_invalidates_cached_context(monkeypatch):
    import minecraft_mod_ai.llama_exact_context as module

    client = _Client()
    identity = {"value": "managed:generation-a"}
    module._MANAGED_CONTEXT_CACHE.clear()
    monkeypatch.setattr(module, "_client", lambda _url: client)
    monkeypatch.setattr(
        module, "_managed_generation_identity", lambda _url: identity["value"]
    )
    live_context_accounting(
        "http://127.0.0.1:8910/v1",
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    identity["value"] = "managed:generation-b"
    live_context_accounting(
        "http://127.0.0.1:8910/v1",
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    assert client.props_calls == 2
