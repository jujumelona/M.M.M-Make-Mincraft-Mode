from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import platform_repair_target_contract as platform_repair


def test_repair_source_limits_remain_bounded_without_mutating_shared_schema():
    from minecraft_mod_ai.model_response_templates import response_schema
    from minecraft_mod_ai.repair_response_contract import repair_response_schema

    bounded = repair_response_schema(8192)
    branches = bounded["properties"]["operations"]["items"]["anyOf"]
    assert branches[1]["properties"]["content"]["maxLength"] == 8192
    assert branches[2]["properties"]["replacements"]["items"]["properties"]["new"]["maxLength"] == 4096
    assert branches[1]["properties"]["expected_sha256"]["pattern"] == "^sha256:[0-9a-f]{64}$"
    assert response_schema("repair")["properties"]["operations"]["items"]["anyOf"][1]["properties"]["content"]["maxLength"] == 256


def test_repair_candidate_is_inert_and_defers_scope_commit() -> None:
    seen: dict[str, object] = {}

    class Router:
        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("repair candidate generation must not author structured JSON text")

        def generate_tool_decision(
            self,
            role,
            messages,
            *,
            tool_name,
            parameters,
            description="",
        ):
            seen["role"] = role
            seen["messages"] = messages
            seen["tool_name"] = tool_name
            seen["parameters"] = parameters
            seen["description"] = description
            return {
                "operations": [
                    {
                        "operation": "create",
                        "path": "src/main/java/PureCandidate.java",
                        "content": "final class PureCandidate {\n" + "    // source context\n" * 25 + "}\n",
                    }
                ]
            }

    class Engine:
        def __init__(self) -> None:
            self.router = Router()
            self.policy = SimpleNamespace(max_patch_bytes=1024 * 1024)
            self._mmm_last_java_paths = ("winner-scope.java",)

        def _request_patch(self, _evidence, _context):
            raise AssertionError("platform repair installer must replace this method")

    module = SimpleNamespace(
        RepairEngine=Engine,
        RepairEngineError=RuntimeError,
        _extract_json=json.loads,
    )
    platform_repair._install_dynamic_patch_request(module)

    adapter = SimpleNamespace(
        minecraft_version="1.21.1",
        loader="fabric",
        yarn_mappings="1.21.1+build.3",
        java_version="21",
        fabric_loader="0.16.10",
        fabric_api="0.116.4+1.21.1",
        fabric_loom="1.9.2",
        gradle="8.10.2",
    )
    token = platform_repair._ACTIVE_REPAIR_TARGET.set(adapter)
    try:
        engine = Engine()
        operations = engine._request_patch(
            {"diagnostics": {"diagnostics": []}},
            {"manifest": {"project_sha256": "sha256:test"}},
        )
    finally:
        platform_repair._ACTIVE_REPAIR_TARGET.reset(token)

    assert operations[0]["path"] == "src/main/java/PureCandidate.java"
    assert seen["role"] == "coder"
    assert seen["tool_name"] == "submit_fixed_template"
    assert seen["parameters"]["additionalProperties"] is False
    assert engine._mmm_last_java_paths == ("winner-scope.java",)
    assert getattr(engine._request_patch, "_mmm_defers_repair_scope_commit", False)
    assert getattr(engine._request_patch, "_mmm_pure_candidate_generation", False)
