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
from test_implementation_ir import graph_project

from minecraft_mod_ai import custom_module_generator as direct
from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_router import ModelRouter


@pytest.mark.parametrize("fault", ["public_api", "estimated_tokens"])
def test_native_graph_correction_reaches_java_execution(tmp_path, monkeypatch, fault):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    module, main = graph_project(tmp_path)
    module.config["implementation_graph_request"]["text"] = (
        "# Trading\n- PlayerCredits owns balances and rejects insufficient purchases.\n"
        "- TradeService runs one purchase at initialization and verifies the remaining balance."
    )
    wallet_api = ["public static int balance()", "public static boolean spend(int amount)"]
    wallet = {"public_api": wallet_api, "activation": False, "estimated_tokens": 800}
    invalid = {**wallet, fault: "not-an-array" if fault == "public_api" else "missing estimate"}
    # Real HTTP/native parsing, including one malformed scalar/array. Only the
    # invalid field is re-requested; identity and behavior never get replayed.
    pages = [
        {"symbol": "PlayerCredits", "kind": "java"},
        {"state_transition": "PlayerCredits owns balances and spends once.",
         "success_condition": "Balance is reduced by price.", "failure_condition": "Insufficient funds leave state unchanged."},
        {"depends_on": []}, invalid, {fault: wallet[fault]},
        {"symbol": "TradeService", "kind": "java"},
        {"state_transition": "Register one startup purchase using PlayerCredits.",
         "success_condition": "Remaining credits equal seven.", "failure_condition": "Unaffordable purchase keeps seven credits."},
        {"depends_on": ["PlayerCredits"]},
        {"public_api": [], "activation": True, "estimated_tokens": 700},
    ]
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
            super().__init__()

        def _generation_adapter(self, role):
            return configs[role], adapters[role]

        def _generation_scope(self, config):
            return nullcontext()

        def generate_text(self, role, messages, **kwargs):
            # Source responses are controlled fixtures too; exercise their real
            # HTTP transport without pretending to run live MCP/RAG research.
            return adapters[role].generate(GenerationRequest(messages=messages))

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
        assert len(requests) == 11
        for payload in requests[:len(pages)]:
            assert len(payload["tools"]) == 1
            assert payload["tools"][0]["function"]["name"] != "compile_implementation_graph"
            assert "nodes" not in payload["tools"][0]["function"]["parameters"]["properties"]
            assert payload["tool_choice"] == "required"
            assert payload["parallel_tool_calls"] is False
        schema = requests[3]["tools"][0]["function"]["parameters"]
        assert list(Draft202012Validator(schema).iter_errors(invalid))
        assert not list(Draft202012Validator(schema).iter_errors(wallet))
        repair_schema = requests[4]["tools"][0]["function"]["parameters"]
        assert repair_schema["required"] == [fault]
        correction = json.loads(next(m["content"] for m in requests[4]["messages"] if m["role"] == "user"))
        assert fault in correction["correct_only"]
        assert set(correction["accepted_fields"]) == set(wallet) - {fault}
        continuation = json.loads(next(m["content"] for m in requests[5]["messages"] if m["role"] == "user"))
        assert list(continuation["work_packet"]["requirements"]) == ["R3"]
        interface_request = json.loads(next(m["content"] for m in requests[8]["messages"] if m["role"] == "user"))
        assert interface_request["dependencies"][0]["public_api"] == wallet_api
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
