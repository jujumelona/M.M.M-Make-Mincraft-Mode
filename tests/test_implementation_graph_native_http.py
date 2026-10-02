"""Exercise host-owned implementation lowering over real coder HTTP, then javac/java.

The server replays controlled native tool responses; implementation planning itself must
perform zero model requests.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_implementation_ir import graph_project
from worksheet_fixtures import row

from minecraft_mod_ai import custom_module_generator as direct
from minecraft_mod_ai import llama_exact_context, llama_lora_runtime
from minecraft_mod_ai import llama_stream_efficiency_contract as streaming
from minecraft_mod_ai.authored_execution_schema import EXECUTION_SECTION_ORDER
from minecraft_mod_ai.authored_structured_design import (
    normalize_structured_sections,
    render_structured_sections,
    structured_sections_sha256,
)
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def test_host_graph_reaches_java_execution_without_planner_http(tmp_path, monkeypatch):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    module, main = graph_project(tmp_path)
    # Keep compiler outputs separate from the generated source transaction.
    class_output = tmp_path.parent / (tmp_path.name + "-compiled")
    structured = {section: row(section) for section in EXECUTION_SECTION_ORDER}
    # The state runtime advertises getState/setState; empty model answers cannot
    # implement that contract. Exercise real host state lowering in this smoke test.
    structured["state_model"]["specification"] = {
        **{name: [] for name in DETAIL_RECORDS["state_model"]},
        "variables": [{"name": "credits", "owner": "Player", "type": "Int",
                       "unit": "credits", "default": "7", "domain": "non-negative"}],
        "inapplicable_concerns": [
            {"concern": name, "reason": "This transport smoke test only stores credits."}
            for name in DETAIL_RECORDS["state_model"] if name != "variables"
        ],
    }
    structured = normalize_structured_sections(structured)
    module.config["implementation_graph_request"].update({
        "text": render_structured_sections(structured),
        "structured_sections": structured,
        "structured_sections_sha256": structured_sections_sha256(structured),
    })
    requests = []

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

            tools = payload.get("tools") or []
            assert tools, "atomic production must use native structured tools only"
            function = tools[0]["function"]
            name = function["name"]
            schema = function["parameters"]
            properties = schema.get("properties", {})
            assembly_request = {}
            for message in reversed(payload["messages"]):
                if message.get("role") != "user":
                    continue
                try:
                    candidate = json.loads(message.get("content") or "")
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(candidate, dict) and isinstance(candidate.get("request"), dict):
                    assembly_request = candidate
                    break
            request = assembly_request.get("request") or {}
            concern_row = request.get("concern") or {}
            concern = (
                str(concern_row.get("name") or "")
                if isinstance(concern_row, dict)
                else str(concern_row or "")
            )
            if not concern:
                concern = str(request.get("host_selected_class") or f"concern_{len(requests)}")
            identifier = re.sub(r"[^A-Za-z0-9_$]", "_", concern)

            if "part" in properties:
                choices = properties["part"].get("enum") or []
                if "fields" in choices:
                    arguments_obj = {"part": "fields"}
                elif "statements" in choices:
                    arguments_obj = {"part": "statements"}
                else:
                    arguments_obj = {"part": "done"}
            elif "statement" in properties:
                arguments_obj = {"statement": f"{identifier}State += 0;"}
            elif "type" in properties and "name" in properties:
                arguments_obj = {
                    "type": "int",
                    "name": f"{identifier}State",
                    "initializer": "0",
                }
            else:
                arguments_obj = {}
                for key in schema.get("required", []):
                    spec = properties[key]
                    if spec.get("enum"):
                        arguments_obj[key] = spec["enum"][0]
                    elif spec.get("type") == "boolean":
                        arguments_obj[key] = False
                    else:
                        arguments_obj[key] = "value"

            arguments = json.dumps(arguments_obj)
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{len(requests)}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
            }
            events = (
                {"choices": [{"delta": delta}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            )

            for event in events:
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

        def _generate_tool_decision_impl(self, role, messages, **kwargs):
            assert role == "coder", "host graph lowering must not call planner tool decisions"
            return super()._generate_tool_decision_impl(role, messages, **kwargs)

    compilations = []

    class JavacRunner:
        def __init__(self, *args):
            pass

        def compile_java(self, root):
            files = list((Path(root) / "src/main/java").rglob("*.java"))
            result = subprocess.run(
                [javac, "-d", str(class_output), *map(str, files)],
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
        assert requests
        assert all(request.get("tools") for request in requests)
        tool_names = {
            request["tools"][0]["function"]["name"]
            for request in requests
        }
        assert tool_names <= {"emit_java_part", "emit_java_statement", "report_java_region_completion"}
        assert "emit_java_part" in tool_names
        assert all("response_format" not in request for request in requests)
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
            "new example.TestMod().onInitialize(); "
            "if (((Number)example.AuthoredStateModel.getState(\"credits\")).intValue() != 7) throw new AssertionError(); "
            "example.AuthoredStateModel.setState(\"credits\", 12); "
            "if (((Number)example.AuthoredStateModel.getState(\"credits\")).intValue() != 12) throw new AssertionError(); } }"
        )
        probe_compile = subprocess.run(
            [
                javac,
                "-cp",
                str(class_output),
                "-d",
                str(class_output),
                str(probe),
                *map(str, (tmp_path / "src/main/java").rglob("*.java")),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert probe_compile.returncode == 0, probe_compile.stderr
        run = subprocess.run(
            [java, "-cp", str(class_output), "Probe"],
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
