"""Exercise host-owned implementation lowering over real coder HTTP, then javac/java.

The server replays one controlled source response; implementation planning itself must
perform zero model requests.
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
from test_implementation_ir import graph_project

from minecraft_mod_ai import custom_module_generator as direct
from minecraft_mod_ai.authored_execution_schema import EXECUTION_SECTION_ORDER, concern_contracts
from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_router import ModelRouter


def test_host_graph_reaches_java_execution_without_planner_http(tmp_path, monkeypatch):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    module, main = graph_project(tmp_path)
    module.config["implementation_graph_request"]["text"] = (
        "## 1. 행동 계약 (behavior_contract)\nBehavior.\n"
        "## 2. 상태 모델 (state_model)\nState.\n"
        "## 3. 알고리즘 (algorithm)\nAlgorithm.\n"
        "## 4. 통합 (integration)\nIntegration.\n"
        "## 5. 권한 및 네트워크 (authority_and_network)\nAuthority.\n"
        "## 6. 지속성 (persistence)\nPersistence.\n"
        "## 7. 자원 및 UI (resources_and_ui)\nResources.\n"
        "## 8. 실패 및 제한 (failure_and_limits)\nFailures.\n"
    )
    atomic_content = (
        "<<<MMM_CONCERN_MEMBERS>>>\n"
        "<<<MMM_CONCERN_INITIALIZE>>>\n"
        "<<<MMM_CONCERN_END>>>"
    )
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            delta = {"content": atomic_content}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in (
                {"choices": [{"delta": delta}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ):
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    configs = {
        role: AdapterConfig(
            role=role,
            adapter="llama_cpp",
            model_id="replayed-native",
            max_new_tokens=4096,
        )
        for role in ("planner", "coder")
    }
    adapters = {role: LlamaCppAdapter(config) for role, config in configs.items()}
    for adapter in adapters.values():
        monkeypatch.setattr(adapter, "_server_url", lambda request: endpoint)
    monkeypatch.setattr(
        llama_exact_context,
        "capacity_safe_payload",
        lambda url, payload, **kwargs: payload,
    )
    monkeypatch.setattr(llama_lora_runtime, "apply_request_lora", lambda *args: None)

    class Router(ModelRouter):
        def __init__(self):
            super().__init__()

        def _generation_adapter(self, role):
            return configs[role], adapters[role]

        def _generation_scope(self, config):
            return nullcontext()

        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            return adapters[role].generate(GenerationRequest(messages=messages))

        def _generate_tool_decision_impl(self, role, messages, **kwargs):
            raise AssertionError("host graph lowering must not call planner tool decisions")

    compilations = []

    class JavacRunner:
        def __init__(self, *args):
            pass

        def compile_java(self, root):
            files = list((Path(root) / "src/main/java").rglob("*.java"))
            result = subprocess.run(
                [javac, "-d", str(tmp_path / "classes"), *map(str, files)],
                capture_output=True,
                text=True,
                check=False,
            )
            compilations.append(result)
            return SimpleNamespace(
                status="PASS" if result.returncode == 0 else "FAIL",
                commands=(),
                error=result.stderr,
            )

    monkeypatch.setattr(direct, "GradleRunner", JavacRunner)
    monkeypatch.setattr(
        direct,
        "adapter_for_target",
        lambda *args: SimpleNamespace(
            minecraft_version="1.21.1",
            loader="fabric",
            java_version=17,
            yarn_mappings="none",
        ),
    )
    try:
        generator = direct.CustomModuleGenerator(Router())
        result = generator.generate(tmp_path, module=module)
        expected_requests = sum(
            len(concern_contracts(section)) for section in EXECUTION_SECTION_ORDER
        ) + len(concern_contracts("integration"))
        assert len(requests) == expected_requests
        assert all(not request.get("tools") for request in requests)
        graph = result["implementation_ir"]
        symbols = {node["symbol"] for node in graph["nodes"]}
        assert symbols == {
            "AuthoredStateModel", "AuthoredBehaviorContract", "AuthoredAlgorithm",
            "AuthoredAuthorityNetwork", "AuthoredPersistence", "AuthoredResourcesUi",
            "AuthoredFailureLimits", "AuthoredIntegration",
        }
        assert all(
            not node["symbol"].startswith("AuthoredGeneric")
            for node in graph["nodes"]
        )
        assert all(c.returncode == 0 for c in compilations)
        assert "AuthoredIntegration.initialize();" in main.read_text()
        assert generator.ensure_generation_live_commit(result, project_root=tmp_path)
        probe = tmp_path / "Probe.java"
        probe.write_text(
            "public class Probe { public static void main(String[] args) { "
            "new example.TestMod().onInitialize(); } }"
        )
        subprocess.run(
            [
                javac,
                "-cp",
                str(tmp_path / "classes"),
                "-d",
                str(tmp_path / "classes"),
                str(probe),
            ],
            check=True,
            capture_output=True,
        )
        run = subprocess.run(
            [java, "-cp", str(tmp_path / "classes"), "Probe"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert run.stdout == ""
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client = streaming._CLIENTS.pop(endpoint, None)
        if client:
            client.close()
