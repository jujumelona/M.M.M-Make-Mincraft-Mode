"""Controlled HTTP/SSE evidence, not a fresh Qwen inference run."""
from __future__ import annotations

import json
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_router import ModelRouter


def test_native_http_assembles_java_without_nested_json_or_repair(monkeypatch):
    requests = []
    responses = [
        {"part": "fields"},
        [
            {"type": "int", "name": "credits", "initializer": "7"},
            {"type": "float", "name": "fuel", "initializer": "100.0"},
            {"type": "boolean", "name": "current_planet_surface", "initializer": True},
            {"type": "int", "name": "reputation_score", "initializer": 0},
        ],
        {"part": "done"},
        {"part": "done"},
        {"part": "classes"},
        {"name": "ShipData"},
        {"part": "done"},
        {"part": "done"},
    ]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(payload)
            name = payload["tools"][0]["function"]["name"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            rows = responses[index]
            if isinstance(rows, dict):
                rows = [rows]
            tool_calls = [
                {
                    "index": call_index,
                    "id": f"call_{index}_{call_index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(row)},
                }
                for call_index, row in enumerate(rows)
            ]
            events = [
                {"choices": [{"delta": {"role": "assistant", "tool_calls": tool_calls}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = AdapterConfig(role="coder", adapter="llama_cpp", model_id="controlled-native", max_new_tokens=4096)
    adapter = LlamaCppAdapter(config)
    monkeypatch.setattr(adapter, "_server_url", lambda _: f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(llama_exact_context, "capacity_safe_payload", lambda _url, payload, **_kw: payload)
    monkeypatch.setattr(llama_lora_runtime, "apply_request_lora", lambda *_: None)
    monkeypatch.setattr(streaming, "_report_server_connection", lambda _: None)

    class Router(ModelRouter):
        def _generation_adapter(self, role):
            assert role == "coder"
            return config, adapter

        def _generation_scope(self, _config):
            return nullcontext()

    try:
        source = _call_atomic_java_region(Router(), [{"role": "user", "content": json.dumps({
            "response_region": "members", "host_selected_class": "AuthoredStateModel",
            "concern": {"name": "variables"},
        })}], output_token_ceiling=None)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert "static int credits = 7;" in source
    assert "static float fuel = 100.0f;" in source
    assert "static boolean current_planet_surface = true;" in source
    assert "static int reputation_score = 0;" in source
    assert "class ShipData" in source
    assert len(requests) == len(responses)
    assert all(r["tools"][0]["function"]["name"] == "emit_java_part" for r in requests)
    assert [r["parallel_tool_calls"] for r in requests] == [
        False, True, False, False, False, True, False, False,
    ]
    assert all(r["tool_choice"] == "required" for r in requests)
    assert all("response_format" not in r for r in requests)
    for request in requests:
        roles = [message["role"] for message in request["messages"]]
        first_user = roles.index("user")
        assert all(role in {"system", "developer"} for role in roles[:first_user])
        assert not any(role in {"system", "developer"} for role in roles[first_user + 1:])
        assert "Call the required function emit_java_part" in request["messages"][0]["content"]
        assert "Repository branch policy" in request["messages"][0]["content"]
        schema = request["tools"][0]["function"]["parameters"]
        assert all(
            value.get("type") not in {"array", "object"}
            for value in schema["properties"].values()
            if isinstance(value.get("type"), str)
        )
