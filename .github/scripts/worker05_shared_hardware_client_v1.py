from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "minecraft_mod_ai/llama_server_hardware_policy.py"
TEST = ROOT / "tests/test_llama_server_hardware_policy.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path.relative_to(ROOT)}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    POLICY,
    "    from .llama_stream_efficiency_contract import _stream_idle_timeout_seconds\n",
    "    from .llama_stream_efficiency_contract import _client, _stream_idle_timeout_seconds\n",
)
replace_once(
    POLICY,
    "        client = httpx.Client()\n",
    "        client = _client(server_url)\n",
)
replace_once(
    POLICY,
    '''    finally:\n        if client is not None:\n            try:\n                client.close()\n            except Exception as close_exc:  # noqa: BLE001 - transport cleanup boundary\n                print(\n                    "llama server: client close failed",\n                    f" error={type(close_exc).__name__}",\n                    flush=True,\n                )\n\n\ndef install(autotune_module: Any) -> None:\n''',
    '''\n\ndef install(autotune_module: Any) -> None:\n''',
)

text = TEST.read_text(encoding="utf-8")
text = text.replace(
    "from minecraft_mod_ai import llama_server_hardware_policy as policy\n",
    "from minecraft_mod_ai import llama_server_hardware_policy as policy\nfrom minecraft_mod_ai import llama_stream_efficiency_contract as stream\n",
    1,
)
old_test = '''def test_strict_server_generate_reuses_one_http_client_without_auxiliary_metrics(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)\n    client = _FakeClient()\n    created: list[_FakeClient] = []\n    fake_httpx = SimpleNamespace(\n        Client=lambda **_kwargs: created.append(client) or client,\n        Timeout=lambda **_kwargs: object(),\n        Limits=lambda **_kwargs: object(),\n    )\n    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)\n    monkeypatch.setattr(\n        policy,\n        "_TELEMETRY_TOTALS",\n        {\n            "prompt_tokens": 0,\n            "output_tokens": 0,\n            "generation_seconds": 0.0,\n            "requests": 0,\n        },\n    )\n\n    class Adapter:\n        _reported_server_url = None\n        config = SimpleNamespace(\n            role="code_generator",\n            model_id="test-model",\n            max_new_tokens=-1,\n        )\n\n    request = SimpleNamespace(\n        messages=({"role": "user", "content": "x"},),\n        response_format="text",\n        tools=(),\n    )\n\n    result = policy._strict_server_generate(\n        Adapter(), request, "http://127.0.0.1:8080/v1"\n    )\n\n    assert result == "ok"\n    assert created == [client]\n    assert client.metrics_gets == 0\n    assert client.stream_calls == 1\n    assert not client.closed\n'''
new_test = '''def test_strict_server_generate_reuses_shared_http_client_without_auxiliary_metrics(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)\n    client = _FakeClient()\n    requested_urls: list[str] = []\n    fake_httpx = SimpleNamespace(\n        Timeout=lambda **_kwargs: object(),\n    )\n    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)\n    monkeypatch.setattr(\n        stream,\n        "_client",\n        lambda server_url: requested_urls.append(server_url) or client,\n    )\n    monkeypatch.setattr(\n        policy,\n        "_TELEMETRY_TOTALS",\n        {\n            "prompt_tokens": 0,\n            "prompt_seconds": 0.0,\n            "output_tokens": 0,\n            "generation_seconds": 0.0,\n            "requests": 0,\n        },\n    )\n\n    class Adapter:\n        _reported_server_url = None\n        config = SimpleNamespace(\n            role="code_generator",\n            model_id="test-model",\n            max_new_tokens=-1,\n        )\n\n    request = SimpleNamespace(\n        messages=({"role": "user", "content": "x"},),\n        response_format="text",\n        tools=(),\n    )\n    server_url = "http://127.0.0.1:8080/v1"\n\n    assert policy._strict_server_generate(Adapter(), request, server_url) == "ok"\n    assert policy._strict_server_generate(Adapter(), request, server_url) == "ok"\n\n    assert requested_urls == [server_url, server_url]\n    assert client.metrics_gets == 0\n    assert client.stream_calls == 2\n    assert not client.closed\n'''
if text.count(old_test) != 1:
    raise SystemExit(f"hardware policy test: expected one old test, found {text.count(old_test)}")
TEST.write_text(text.replace(old_test, new_test, 1), encoding="utf-8")

# One-shot staging artifacts must not remain in production main.
(ROOT / ".github/workflows/worker05-shared-hardware-client-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
