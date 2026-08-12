from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import platform_repair_target_contract as platform_repair


def test_repair_candidate_is_inert_and_defers_scope_commit() -> None:
    seen: dict[str, object] = {}

    class Router:
        def generate_text(self, role, messages, **kwargs):
            seen["role"] = role
            seen["messages"] = messages
            seen.update(kwargs)
            return json.dumps(
                {
                    "operations": [
                        {
                            "operation": "create",
                            "path": "src/main/java/PureCandidate.java",
                            "content": "final class PureCandidate {}",
                        }
                    ]
                }
            )

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
    assert seen["response_format"] == "json"
    assert seen["enable_tools"] is False
    assert engine._mmm_last_java_paths == ("winner-scope.java",)
    assert getattr(engine._request_patch, "_mmm_defers_repair_scope_commit", False)
    assert getattr(engine._request_patch, "_mmm_pure_candidate_generation", False)
