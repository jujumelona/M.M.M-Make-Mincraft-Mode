"""Exercise admission correction over real HTTP/native parsing, then javac/java.

The server replays controlled responses; this is not a live-model or Fabric test.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_implementation_ir import graph_project, node

from minecraft_mod_ai import custom_module_generator as direct
from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_router import ModelRouter


@pytest.mark.parametrize("fault", ["public_api", "resource_path"])
def test_native_graph_correction_reaches_java_execution(tmp_path, monkeypatch, fault):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    module, main = graph_project(tmp_path)
    wallet = node(refs=["R1", "R2", "R3", "R4", "R5"], api=["public static int balance()", "public static boolean spend(int amount)"])
    trade = node("TradeService", refs=["R6"], dependencies=["PlayerCredits"], activation=True,
                 api=["public static void initialize()"])
    invalid = {**wallet, fault: [] if fault == "public_api" else "src/main/java/example/PlayerCredits.java"}
    # Premature done=true must request the remaining design requirement, not abort.
    pages = [{"nodes": [invalid], "done": True}, {"nodes": [wallet], "done": True},
             {"nodes": [trade], "done": True}]
    bodies = [
        ("package example;\npublic final class PlayerCredits { private static int credits=10; "
        "public static int balance() { return credits; } "
        "public static boolean spend(int amount) { if(amount<0 || credits<amount) return false; credits-=amount; return true; } }"),
        ('package example;\npublic final class TradeService { public static void initialize() { '
        'if(!PlayerCredits.spend(3) || PlayerCredits.balance()!=7 || PlayerCredits.spend(8) || PlayerCredits.balance()!=7) '
        'throw new AssertionError("transaction"); System.out.print("PASS"); } }'),
    ]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(payload)
            if index < len(pages):
                name = payload["tools"][0]["function"]["name"]
                delta = {"tool_calls": [{"index": 0, "id": f"graph_{index}", "type": "function",
                                         "function": {"name": name, "arguments": json.dumps(pages[index])}}]}
                finish = "tool_calls"
            else:
                delta = {"content": json.dumps({"content": bodies[index - len(pages)], "summary": "implemented"})}
                finish = "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in ({"choices": [{"delta": delta}]}, {"choices": [{"delta": {}, "finish_reason": finish}]}):
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    configs = {role: AdapterConfig(role=role, adapter="llama_cpp", model_id="replayed-native",
                                    max_new_tokens=8192 if role == "planner" else 4096)
               for role in ("planner", "coder")}
    adapters = {role: LlamaCppAdapter(config) for role, config in configs.items()}
    for adapter in adapters.values():
        monkeypatch.setattr(adapter, "_server_url", lambda request: endpoint)
    # No local model/GPU: substitute capacity/LoRA discovery only, retaining production
    # payloads, HTTP streaming, native schema validation, router and executor.
    monkeypatch.setattr(llama_exact_context, "capacity_safe_payload", lambda url, payload, **kwargs: payload)
    monkeypatch.setattr(llama_lora_runtime, "apply_request_lora", lambda *args: None)

    class Router(ModelRouter):
        def __init__(self):
            self._agent_require_fresh_evidence = False

        def _generation_adapter(self, role):
            return configs[role], adapters[role]

        def _generation_scope(self, config):
            return nullcontext()

    compilations = []

    class JavacRunner:
        def __init__(self, *args):
            pass

        def compile_java(self, root):
            files = list((Path(root) / "src/main/java").rglob("*.java"))
            result = subprocess.run([javac, "-d", str(tmp_path / "classes"), *map(str, files)], capture_output=True, text=True, check=False)
            compilations.append(result)
            return SimpleNamespace(status="PASS" if result.returncode == 0 else "FAIL", commands=(), error=result.stderr)

    monkeypatch.setattr(direct, "GradleRunner", JavacRunner)
    monkeypatch.setattr(direct, "adapter_for_target", lambda *args: SimpleNamespace(
        minecraft_version="1.21.1", loader="fabric", java_version=17, yarn_mappings="none"))
    try:
        generator = direct.CustomModuleGenerator(Router())
        result = generator.generate(tmp_path, module=module)
        assert len(requests) == 5
        for payload in requests[:3]:
            assert len(payload["tools"]) == 1
            assert payload["tools"][0]["function"]["name"] == "compile_implementation_graph"
            assert payload["tool_choice"] == "required"
            assert payload["parallel_tool_calls"] is False
        schema = requests[0]["tools"][0]["function"]["parameters"]
        assert list(Draft202012Validator(schema).iter_errors(pages[0]))
        assert not list(Draft202012Validator(schema).iter_errors(pages[1]))
        correction = json.loads(next(m["content"] for m in requests[1]["messages"] if m["role"] == "user"))
        assert fault in json.dumps(correction["validation_feedback"])
        assert correction["validation_feedback"]["rejected_page"] == pages[0]
        continuation = json.loads(next(m["content"] for m in requests[2]["messages"] if m["role"] == "user"))
        assert continuation["remaining_requirements"] == ["R6"]
        assert continuation["accepted_nodes"][0]["symbol"] == "PlayerCredits"
        assert all(c.returncode == 0 for c in compilations)
        assert "TradeService.initialize();" in main.read_text()
        assert generator.ensure_generation_live_commit(result, project_root=tmp_path)
        probe = tmp_path / "Probe.java"
        probe.write_text("public class Probe { public static void main(String[] args) { new example.TestMod().onInitialize(); } }")
        subprocess.run([javac, "-cp", str(tmp_path / "classes"), "-d", str(tmp_path / "classes"), str(probe)], check=True, capture_output=True)
        run = subprocess.run([java, "-cp", str(tmp_path / "classes"), "Probe"], check=True, capture_output=True, text=True)
        assert run.stdout == "PASS"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client = streaming._CLIENTS.pop(endpoint, None)
        if client:
            client.close()
