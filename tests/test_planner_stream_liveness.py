"""Replay the reported healthy >300s plan through the actual HTTP SSE path."""
from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_completion_liveness_contract as liveness
from minecraft_mod_ai import (
    llama_exact_context,
    llama_lora_runtime,
    llama_server_hardware_policy,
)
from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_router import ModelRouter


def test_complete_planner_finishes_healthy_8192_budget_stream_after_300_seconds(monkeypatch):
    for key in ("MMM_LLAMA_COMPLETION_WALL_TIMEOUT_SECONDS", "MMM_LLAMA_TOOL_COMPLETION_WALL_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    # Only the liveness clock is accelerated. HTTPX and the local TCP server retain
    # real clocks. 4.2 real seconds replay a 420s decode, without a GPU or sleeps of minutes.
    real_monotonic = time.monotonic
    monkeypatch.setattr(liveness, "time", SimpleNamespace(monotonic=lambda: real_monotonic() * 100))
    requests = []
    chunks = ["# behavior_contract\n"] + [f"Requirement {i}.\n" for i in range(28)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for index, content in enumerate(chunks):
                    if index:
                        time.sleep(0.15)
                    event = {"choices": [{"delta": {"content": content}}]}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    config = AdapterConfig(role="planner", adapter="llama_cpp", model_id="replayed-planner", max_new_tokens=8192)
    adapter = LlamaCppAdapter(config)
    monkeypatch.setattr(adapter, "_server_url", lambda request: endpoint)
    monkeypatch.setattr(llama_exact_context, "capacity_safe_payload", lambda url, payload, **kwargs: payload)
    monkeypatch.setattr(llama_lora_runtime, "apply_request_lora", lambda *args: None)
    monkeypatch.setattr(llama_server_hardware_policy, "_server_payload", lambda adapter, request: {
        "model": config.model_id, "messages": list(request.messages), "max_tokens": 8192,
    })

    class Router(ModelRouter):
        def __init__(self):
            self._agent_require_fresh_evidence = False

        def _generation_adapter(self, role):
            return config, adapter

        def _tools_enabled(self, **kwargs):
            return False

        def _generation_scope(self, config):
            return nullcontext()

    try:
        plan = CompleteGameDesignPlanner(Router()).plan("Write the space trading design.")
        assert plan.text == "".join(chunks).strip()
        assert len(requests) == 1
        assert requests[0]["max_tokens"] == 8192
        assert not requests[0].get("tools")
        assert requests[0]["return_progress"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client = streaming._CLIENTS.pop(endpoint, None)
        if client is not None:
            client.close()


@pytest.mark.parametrize("payload", [
    {"max_tokens": 8192}, {"max_completion_tokens": 8192}, {"n_predict": 8192},
    {"max_tokens": 8192, "tools": [{"type": "function"}]},
])
def test_finite_output_uses_progress_liveness_not_an_implicit_wall_deadline(monkeypatch, payload):
    monkeypatch.delenv("MMM_LLAMA_COMPLETION_WALL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MMM_LLAMA_TOOL_COMPLETION_WALL_TIMEOUT_SECONDS", raising=False)
    assert liveness._completion_wall_timeout_seconds(payload) is None


def test_finite_long_request_still_aborts_semantic_stall_with_keepalives(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(liveness, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def lines():
        for at in range(0, 301, 15):
            clock[0] = float(at)
            yield 'data: {"choices":[{"delta":{"content":"x"}}]}'
        for at in range(315, 436, 15):
            clock[0] = float(at)
            yield ": ping"

    response = liveness._ProgressCheckedResponse(
        SimpleNamespace(iter_lines=lines), 120, request_id="stalled-after-progress", started_at=0,
        wall_seconds=liveness._completion_wall_timeout_seconds({"max_tokens": 8192}),
    )
    with pytest.raises(liveness.LlamaSemanticProgressTimeout, match="no semantic"):
        list(response.iter_lines())
    assert clock[0] == 420.0


@pytest.mark.parametrize("tools,env_name,fallback", [
    ([], "MMM_LLAMA_COMPLETION_WALL_TIMEOUT_SECONDS", 300.0),
    ([{"type": "function"}], "MMM_LLAMA_TOOL_COMPLETION_WALL_TIMEOUT_SECONDS", 600.0),
])
def test_unbounded_decode_keeps_fallback_and_explicit_deadline_is_honored(monkeypatch, tools, env_name, fallback):
    monkeypatch.delenv(env_name, raising=False)
    assert liveness._completion_wall_timeout_seconds({"tools": tools}) == fallback
    assert liveness._completion_wall_timeout_seconds({"tools": tools, "max_tokens": -1}) == fallback
    monkeypatch.setenv(env_name, "45")
    assert liveness._completion_wall_timeout_seconds({"tools": tools, "max_tokens": 8192}) == 45
    monkeypatch.setenv(env_name, "nan")
    with pytest.raises(ValueError, match="positive finite"):
        liveness._completion_wall_timeout_seconds({"tools": tools, "max_tokens": 8192})


def test_explicit_deadline_is_not_replayed_as_a_semantic_stall():
    import httpx

    calls = []

    class Client:
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            raise liveness.LlamaCompletionDeadlineExceeded("explicit deadline")

    with pytest.raises(liveness.LlamaCompletionDeadlineExceeded):
        liveness._post_completion_with_transport_replay(
            Client(), "http://localhost/chat/completions", payload={"max_tokens": 8192},
            timeout=None, httpx_module=httpx, request_id="explicit-deadline",
        )
    assert len(calls) == 1
