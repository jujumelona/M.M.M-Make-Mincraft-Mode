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


def test_native_http_schema_rejection_preserves_siblings_and_repairs_one_component(monkeypatch):
    requests = []
    responses = [
        {"fields": [{"type": "int", "name": "credits", "initializer": "7"}],
         "classes": [{"name": "AuthoredStateModel"}]},
        {"value": "ShipData"},
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
            events = [
                {"choices": [{"delta": {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": f"call_{index}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(responses[index])},
                }]}}]},
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
    assert "class ShipData" in source
    assert [r["tools"][0]["function"]["name"] for r in requests] == [
        "emit_java_structure", "repair_java_component",
    ]
    assert all(r["parallel_tool_calls"] is False for r in requests)
    assert all(r["tool_choice"] == "required" for r in requests)
    assert all("response_format" not in r for r in requests)
    repair_schema = requests[1]["tools"][0]["function"]["parameters"]
    assert set(repair_schema["properties"]) == {"value"}
