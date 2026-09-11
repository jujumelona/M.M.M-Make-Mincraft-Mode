from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai.generation_accuracy_contract import (
    GenerationAccuracyError,
    install_inner,
    install_outer,
)


def _messages(path: str = "src/main/resources/data/example/value.json") -> list[dict[str, str]]:
    request = {
        "phase": "implement_module",
        "checkpoint": {"resumed": False},
        "module": {
            "module_id": "example",
            "kind": "custom_java",
            "evidence_task": {
                "task_id": "task.example",
                "coder_execution_contract": {
                    "schema_version": "mmm/atomic-coder-step",
                    "task_ref": "task.example",
                    "targets": [
                        {
                            "path": path,
                            "locator": path,
                            "symbol": "",
                            "operation": "create_or_replace",
                        }
                    ],
                    "protected_boundaries": {"writable_paths": [path]},
                    "step": {
                        "index": 1,
                        "count": 1,
                        "obligation": "write exact resource",
                        "target_refs": [path],
                        "must_provide": ["valid resource"],
                        "done_when": "resource exists and is structurally valid",
                    },
                },
            },
        },
    }
    return [
        {"role": "system", "content": "coder"},
        {"role": "user", "content": json.dumps(request)},
    ]


def _router_module(generate_impl: Any) -> Any:
    class Router:
        def __init__(self, workspace_root):
            self._agent_workspace_root = workspace_root
            self.calls = 0

        def generate_text(self, role, messages, *args, **kwargs):
            self.calls += 1
            return generate_impl(self, role, messages, *args, **kwargs)

    return SimpleNamespace(ModelRouter=Router)


def test_atomic_accuracy_accepts_valid_owned_target(tmp_path):
    target = tmp_path / "src/main/resources/data/example/value.json"

    def generate(router, role, messages, *args, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"value":1}\n', encoding="utf-8")
        return '{"summary":"implemented"}'

    module = _router_module(generate)
    install_inner(module)
    router = module.ModelRouter(tmp_path)

    result = router.generate_text("coder", _messages(), tool_stage="generation")

    assert json.loads(result)["summary"] == "implemented"
    assert router.calls == 1


def test_atomic_accuracy_repairs_invalid_json_locally(tmp_path):
    target = tmp_path / "src/main/resources/data/example/value.json"

    def generate(router, role, messages, *args, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        has_repair = any(
            "HOST GENERATION-ACCURACY GATE FAILED" in str(message.get("content") or "")
            for message in messages
        )
        target.write_text(
            '{"value":2}\n' if has_repair else '{"value":',
            encoding="utf-8",
        )
        return '{"summary":"repaired"}' if has_repair else '{"summary":"first pass"}'

    module = _router_module(generate)
    install_inner(module)
    router = module.ModelRouter(tmp_path)

    result = router.generate_text("coder", _messages(), tool_stage="generation")

    assert json.loads(result)["summary"] == "repaired"
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}
    assert router.calls == 2


def test_atomic_accuracy_fails_closed_on_outside_target_mutation(tmp_path):
    target = tmp_path / "src/main/resources/data/example/value.json"
    outside = tmp_path / "src/main/resources/data/example/unowned.json"

    def generate(router, role, messages, *args, **kwargs):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"value":1}\n', encoding="utf-8")
        outside.write_text('{"unowned":true}\n', encoding="utf-8")
        return '{"summary":"implemented"}'

    module = _router_module(generate)
    install_inner(module)
    router = module.ModelRouter(tmp_path)

    with pytest.raises(GenerationAccuracyError, match="OUTSIDE_TARGET_MUTATION"):
        router.generate_text("coder", _messages(), tool_stage="generation")
    assert router.calls == 1


def test_outer_accuracy_normalizes_atomic_aggregate_summary(tmp_path):
    def generate(router, role, messages, *args, **kwargs):
        return "atomic step 1/2: first\natomic step 2/2: second"

    module = _router_module(generate)
    install_outer(module)
    router = module.ModelRouter(tmp_path)

    result = router.generate_text("coder", _messages())

    assert json.loads(result) == {
        "summary": "atomic step 1/2: first\natomic step 2/2: second"
    }
