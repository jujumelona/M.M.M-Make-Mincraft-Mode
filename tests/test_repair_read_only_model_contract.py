from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.platform_repair_target_contract import _ACTIVE_REPAIR_TARGET
from minecraft_mod_ai.repair_engine import RepairEngine


class _Router:
    profile = "test"

    def __init__(self) -> None:
        self.bound = None
        self.calls = []

    def bind_agent_workspace(self, root, *, require_fresh_evidence):
        self.bound = (root, require_fresh_evidence)

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return json.dumps(
            {
                "operations": [
                    {
                        "operation": "create",
                        "path": "src/main/java/example/Fix.java",
                        "content": "final class Fix {}",
                    }
                ]
            }
        )


def test_repair_model_returns_inert_patch_proposals_without_source_tools() -> None:
    router = _Router()
    engine = RepairEngine.__new__(RepairEngine)
    engine.router = router
    engine.policy = SimpleNamespace(max_patch_bytes=64 * 1024)
    target = SimpleNamespace(
        minecraft_version="1.21.1",
        loader="fabric",
        yarn_mappings="1.21.1+build.3",
        java_version="21",
        fabric_loader="0.16.10",
        fabric_api="0.115.0+1.21.1",
        fabric_loom="1.9-SNAPSHOT",
        gradle="8.12",
    )

    token = _ACTIVE_REPAIR_TARGET.set(target)
    try:
        operations = engine._request_patch(
            {"passed": False, "diagnostics": {}, "build": {"status": "FAIL"}},
            {"manifest": {}, "relevant": []},
        )
    finally:
        _ACTIVE_REPAIR_TARGET.reset(token)

    assert operations[0]["operation"] == "create"
    assert len(router.calls) == 1
    role, messages, kwargs = router.calls[0]
    assert role == "coder"
    assert kwargs["response_format"] == "json"
    assert kwargs["enable_tools"] is False
    assert "tool_stage" not in kwargs
    assert router.bound is None
    prompt = json.loads(messages[-1]["content"])
    assert prompt["target"]["minecraft_version"] == "1.21.1"
    assert prompt["target"]["mappings"] == "1.21.1+build.3"
