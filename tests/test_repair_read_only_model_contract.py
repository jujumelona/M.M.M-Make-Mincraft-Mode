from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai import production_tools
from minecraft_mod_ai.repair_engine import RepairEngine, _ACTIVE_REPAIR_PROJECT_INDEX


class _Index:
    @staticmethod
    def manifest_receipt() -> dict:
        return {"sha256": "sha256:" + "1" * 64}


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


class _ToolService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def index_project_rag(self, *args, **kwargs):
        return {"status": "PASS"}


def test_repair_model_can_retrieve_but_cannot_own_source_writes(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    (root / ".minecraft_ai").mkdir(parents=True)
    (root / ".minecraft_ai" / "platform-lock.json").write_text(
        json.dumps(
            {
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "yarn_mappings": "1.21.1+build.3",
                "java_version": "21",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(production_tools, "ProductionToolService", _ToolService)

    router = _Router()
    engine = RepairEngine.__new__(RepairEngine)
    engine.router = router
    engine.policy = SimpleNamespace(max_patch_bytes=64 * 1024)

    token = _ACTIVE_REPAIR_PROJECT_INDEX.set((root, _Index()))
    try:
        operations = engine._request_patch(
            {"passed": False, "diagnostics": {}, "build": {"status": "FAIL"}},
            {"manifest": {}, "relevant": []},
        )
    finally:
        _ACTIVE_REPAIR_PROJECT_INDEX.reset(token)

    assert operations[0]["operation"] == "create"
    assert len(router.calls) == 1
    role, _messages, kwargs = router.calls[0]
    assert role == "coder_safe"
    assert kwargs["tool_stage"] == "quality"
    assert kwargs["response_format"] == "json"
    assert router.bound == (root.parent, True)
